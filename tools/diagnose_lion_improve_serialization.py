#!/usr/bin/env python3
import _init_path  # noqa: F401
import argparse
import copy
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.backbones_3d.lion_improve import (
    build_geometry_order_from_coords,
    build_group_tokens_from_mapping,
    build_serialization_graph_context,
    build_topology_order_from_context,
    reverse_order_within_batches,
    summarize_object_fragmentation,
    summarize_serialized_groups,
)
from pcdet.utils import common_utils


DEFAULT_ORDERS = [
    'x',
    'y',
    'bev_z',
    'bev_z_t',
    'bev_h',
    'bev_h_t',
    'bev_z_25d',
    'bev_z_25d_t',
    'bev_h_25d',
    'bev_h_25d_t',
    'z3d',
    'z3d_t',
    'h3d',
    'h3d_t',
    'topology',
    'topology_rev',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Compute LION_improve serialization diagnostics on Ped20 train/val.')
    parser.add_argument(
        '--cfg_file',
        default='tools/cfgs/kitti_models/second_with_lion_mamba_64dim_sbsd/second_with_lion_mamba_64dim_sbsd_baseline_fromscratch.yaml',
    )
    parser.add_argument('--data_path', default='/root/autodl-tmp/kitti-offical')
    parser.add_argument(
        '--output_dir',
        default='output/LION_output/cfgs/kitti_models/LION_improve/subset/serialization_diagnostics',
    )
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=0, help='0 means full split')
    parser.add_argument('--orders', default=','.join(DEFAULT_ORDERS))
    parser.add_argument('--knn_k', type=int, default=8)
    parser.add_argument('--knn_max_queries_per_batch', type=int, default=512)
    parser.add_argument('--boundary_connectivity', type=int, default=26)
    parser.add_argument(
        '--stage_filter',
        default='linear_1_enc1,linear_2_enc1,linear_3_enc1,linear_4_enc1,linear_out',
        help='Comma-separated LIONLayer debug names to diagnose, or "all". Default is one representative per stage.',
    )
    parser.add_argument('--disable_knn', action='store_true')
    parser.add_argument('--disable_boundary', action='store_true')
    parser.add_argument('--disable_object_fragmentation', action='store_true')
    parser.add_argument('--train_split', default='train_sbsd_fromscratch_pct20')
    parser.add_argument('--train_info', default='kitti_infos_train_sbsd_fromscratch_pct20.pkl')
    parser.add_argument('--val_split', default='val')
    parser.add_argument('--val_info', default='kitti_infos_val.pkl')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def force_serialization_only_baseline(model_cfg):
    lion_improve_cfg = model_cfg.BACKBONE_3D.get('LION_IMPROVE', None)
    if lion_improve_cfg is None:
        return
    lion_improve_cfg.ENABLED = False
    if lion_improve_cfg.get('SERIALIZATION', None) is not None:
        lion_improve_cfg.SERIALIZATION.ENABLED = False
        lion_improve_cfg.SERIALIZATION.EXECUTION_MODE = 'serial'
    if lion_improve_cfg.get('DIAGNOSTICS', None) is not None:
        lion_improve_cfg.DIAGNOSTICS.ENABLED = False


def make_split_cfg(base_cfg, split_name, info_path):
    split_cfg = copy.deepcopy(base_cfg)
    split_cfg.DATA_CONFIG.DATA_PATH = str(base_cfg.DATA_CONFIG.DATA_PATH)
    split_cfg.DATA_CONFIG.DATA_SPLIT.test = split_name
    split_cfg.DATA_CONFIG.INFO_PATH.test = [info_path]
    return split_cfg


def stage_stride_xyz(debug_name):
    if debug_name.startswith('linear_1'):
        return [1, 1, 1]
    if debug_name.startswith('linear_2'):
        return [1, 1, 2]
    if debug_name.startswith('linear_3'):
        return [1, 1, 4]
    if debug_name.startswith('linear_4'):
        return [1, 1, 8]
    if debug_name.startswith('linear_out'):
        return [1, 1, 16]
    return [1, 1, 1]


class SerializationDiagnosticCollector:
    def __init__(self, orders, args, dataset):
        self.orders = list(orders)
        self.args = args
        self.dataset = dataset
        self.records = []
        self.current_gt_boxes = None
        self.current_split = None
        self.current_batch_idx = -1
        self.stage_filter = None
        if str(args.stage_filter).lower() != 'all':
            self.stage_filter = {item.strip() for item in args.stage_filter.split(',') if item.strip()}
        self.graph_cfg = {
            'GRAPH': {
                'NEIGHBORHOOD': 26,
                'SIGMA_P': 1.5,
                'SIGMA_RHO': 4.0,
                'RESPONSE_DETACH': True,
                'RESPONSE_GAMMA': 0.25,
                'DENSITY_RANGE_ALPHA': 0.001,
            },
            'SERIALIZATION': {
                'HEAT_RANK_ITERS': 2,
                'HEAT_RANK_ALPHA': 0.65,
            },
        }

    def hook(self, module, inputs):
        if len(inputs) == 0:
            return
        x = inputs[0]
        if not hasattr(x, 'indices') or not hasattr(x, 'features'):
            return
        if self.stage_filter is not None and module.debug_name not in self.stage_filter:
            return
        mappings = module.window_partition(x.indices, x.batch_size, x.spatial_shape)
        order_map = {}
        if 'x' in self.orders:
            order_map['x'] = mappings['x']
        if 'y' in self.orders:
            order_map['y'] = mappings['y']
        for order_name in self.orders:
            if order_name in ('x', 'y', 'topology', 'topology_rev'):
                continue
            order_map[order_name] = build_geometry_order_from_coords(
                coords=x.indices,
                sparse_shape=x.spatial_shape,
                window_shape=module.window_shape,
                shift=module.window_partition.shift,
                order_name=order_name,
            )

        if 'topology' in self.orders or 'topology_rev' in self.orders:
            context = build_serialization_graph_context(
                coords=x.indices,
                features=x.features,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size,
                cfg=self.graph_cfg,
            )
            topology_order = build_topology_order_from_context(
                context=context,
                fallback_order=mappings['x'],
                cfg=self.graph_cfg,
            )
            if 'topology' in self.orders:
                order_map['topology'] = topology_order
            if 'topology_rev' in self.orders:
                order_map['topology_rev'] = reverse_order_within_batches(topology_order, x.indices, x.batch_size)

        for order_name, order in order_map.items():
            groups, valid = build_group_tokens_from_mapping(
                order=order,
                flat2win=mappings['flat2win'],
                coords=x.indices,
                batch_size=x.batch_size,
                group_size=module.group_size,
            )
            prefix = f'{module.debug_name}/{order_name}'
            metrics = summarize_serialized_groups(
                coords=x.indices,
                group_tokens=groups,
                valid=valid,
                knn_k=0 if self.args.disable_knn else self.args.knn_k,
                knn_max_queries_per_batch=self.args.knn_max_queries_per_batch,
                boundary_connectivity=0 if self.args.disable_boundary else self.args.boundary_connectivity,
                prefix=prefix,
            )
            if not self.args.disable_object_fragmentation:
                metrics.update(summarize_object_fragmentation(
                    coords=x.indices,
                    group_tokens=groups,
                    valid=valid,
                    gt_boxes=self.current_gt_boxes,
                    voxel_size=getattr(self.dataset, 'voxel_size', None),
                    point_cloud_range=getattr(self.dataset, 'point_cloud_range', None),
                    stride_xyz=stage_stride_xyz(module.debug_name),
                    prefix=prefix,
                ))
            record = {
                'split': self.current_split,
                'batch_idx': self.current_batch_idx,
                'stage': module.debug_name,
                'order': order_name,
                'num_voxels': int(x.indices.shape[0]),
            }
            short_prefix = f'{prefix}/'
            for key, value in metrics.items():
                if key.startswith(short_prefix):
                    record[key[len(short_prefix):]] = float(value)
            self.records.append(record)


def register_hooks(model, collector):
    handles = []
    for module in model.modules():
        if hasattr(module, 'window_partition') and hasattr(module, 'group_size') and hasattr(module, 'debug_name'):
            handles.append(module.register_forward_pre_hook(collector.hook))
    return handles


def run_split(split_name, split_cfg, args, orders, output_dir):
    logger = common_utils.create_logger(output_dir / f'log_diag_{split_name}.txt', rank=0)
    dataset, loader, _ = build_dataloader(
        dataset_cfg=split_cfg.DATA_CONFIG,
        class_names=split_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
        seed=args.seed,
    )
    model = build_network(model_cfg=split_cfg.MODEL, num_class=len(split_cfg.CLASS_NAMES), dataset=dataset)
    model.cuda().eval()
    collector = SerializationDiagnosticCollector(orders=orders, args=args, dataset=dataset)
    handles = register_hooks(model.backbone_3d, collector)
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            load_data_to_gpu(batch_dict)
            collector.current_split = split_name
            collector.current_batch_idx = batch_idx
            collector.current_gt_boxes = batch_dict.get('gt_boxes', None)
            batch_dict = model.vfe(batch_dict)
            model.backbone_3d(batch_dict)
            if (batch_idx + 1) % 25 == 0:
                logger.info(
                    f'{split_name}: processed {batch_idx + 1} batches, '
                    f'records={len(collector.records)}, elapsed={time.time() - start_time:.1f}s'
                )
    for handle in handles:
        handle.remove()
    logger.info(
        f'{split_name}: finished batches={collector.current_batch_idx + 1}, '
        f'records={len(collector.records)}, elapsed={time.time() - start_time:.1f}s'
    )
    return collector.records


def aggregate_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record['split'], record['stage'], record['order'])].append(record)
    rows = []
    for (split, stage, order), items in sorted(grouped.items()):
        row = {'split': split, 'stage': stage, 'order': order, 'num_records': len(items)}
        metric_names = sorted({
            key for item in items for key in item.keys()
            if key not in ('split', 'batch_idx', 'stage', 'order')
        })
        for metric in metric_names:
            values = [float(item[metric]) for item in items if metric in item]
            if values:
                row[metric] = float(np.mean(values))
        rows.append(row)
    return rows


def write_outputs(output_dir, records, aggregate_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json = output_dir / 'serialization_diagnostics_records.json'
    raw_json.write_text(json.dumps(records, indent=2), encoding='utf-8')
    agg_json = output_dir / 'serialization_diagnostics_summary.json'
    agg_json.write_text(json.dumps(aggregate_rows, indent=2), encoding='utf-8')
    csv_path = output_dir / 'serialization_diagnostics_summary.csv'
    fieldnames = sorted({key for row in aggregate_rows for key in row.keys()})
    front = ['split', 'stage', 'order', 'num_records']
    fieldnames = front + [key for key in fieldnames if key not in front]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)
    return raw_json, agg_json, csv_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = Path(__file__).resolve().parent
    cfg_path = Path(args.cfg_file)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    try:
        cfg_file_for_loader = str(cfg_path.relative_to(tools_dir))
    except ValueError:
        cfg_file_for_loader = str(cfg_path)
    old_cwd = os.getcwd()
    os.chdir(str(tools_dir))
    try:
        cfg_from_yaml_file(cfg_file_for_loader, cfg)
    finally:
        os.chdir(old_cwd)
    cfg.TAG = Path(cfg_file_for_loader).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file_for_loader.split('/')[1:-1])
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    cfg.DATA_CONFIG.DATA_PATH = args.data_path
    force_serialization_only_baseline(cfg.MODEL)

    orders = [item.strip() for item in args.orders.split(',') if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = copy.deepcopy(cfg)
    split_specs = [
        ('train', args.train_split, args.train_info),
        ('val', args.val_split, args.val_info),
    ]
    all_records = []
    for label, split_name, info_path in split_specs:
        split_cfg = make_split_cfg(base_cfg, split_name=split_name, info_path=info_path)
        records = run_split(label, split_cfg, args, orders, output_dir)
        all_records.extend(records)
    aggregate_rows = aggregate_records(all_records)
    _, _, csv_path = write_outputs(output_dir, all_records, aggregate_rows)
    print(f'lion_improve_serialization_diagnostics_done records={len(all_records)} summary={csv_path}')


if __name__ == '__main__':
    main()
