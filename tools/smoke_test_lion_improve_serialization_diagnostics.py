#!/usr/bin/env python3
import _init_path  # noqa: F401
import torch

from pcdet.models.backbones_3d.lion_backbone_one_stride import FlattenedWindowMapping
from pcdet.models.backbones_3d.lion_improve import (
    build_geometry_order_from_coords,
    build_group_tokens_from_mapping,
    build_serialization_graph_context,
    build_topology_order_from_context,
    reverse_order_within_batches,
    summarize_object_fragmentation,
    summarize_serialized_groups,
)


def assert_finite(metrics, name):
    for key, value in metrics.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise AssertionError(f'{name}: non-finite metric {key}={value}')


def make_coords(device):
    coords = []
    for batch in range(2):
        for z in range(4):
            for y in range(4):
                x = (z * 3 + y * 2 + batch) % 12
                coords.append([batch, z, y, x])
                if x + 1 < 12:
                    coords.append([batch, z, y, x + 1])
    return torch.tensor(coords, device=device, dtype=torch.long)


def build_orders(coords, sparse_shape, window_shape, batch_size, mappings):
    features = torch.linspace(0.0, 1.0, coords.shape[0] * 4, device=coords.device).view(coords.shape[0], 4)
    context = build_serialization_graph_context(
        coords=coords,
        features=features,
        spatial_shape=sparse_shape,
        batch_size=batch_size,
        cfg={'GRAPH': {'NEIGHBORHOOD': 26, 'SIGMA_P': 1.5, 'SIGMA_RHO': 4.0, 'RESPONSE_DETACH': True}},
    )
    topology = build_topology_order_from_context(context, fallback_order=mappings['x'], cfg={'SERIALIZATION': {'HEAT_RANK_ITERS': 2}})
    return {
        'x': mappings['x'],
        'y': mappings['y'],
        'bev_z': build_geometry_order_from_coords(coords, sparse_shape, window_shape, True, 'bev_z'),
        'bev_h_25d': build_geometry_order_from_coords(coords, sparse_shape, window_shape, True, 'bev_h_25d'),
        'h3d_t': build_geometry_order_from_coords(coords, sparse_shape, window_shape, True, 'h3d_t'),
        'topology': topology,
        'topology_rev': reverse_order_within_batches(topology, coords, batch_size=batch_size),
    }


def run_device(device):
    coords = make_coords(device)
    sparse_shape = [8, 8, 16]
    window_shape = [4, 4, 2]
    group_size = 8
    batch_size = 2
    mapping = FlattenedWindowMapping(window_shape, group_size, shift=True)
    mappings = mapping(coords, batch_size=batch_size, sparse_shape=sparse_shape)
    orders = build_orders(coords, sparse_shape, window_shape, batch_size, mappings)

    gt_boxes = torch.zeros((batch_size, 2, 8), device=device, dtype=torch.float32)
    gt_boxes[0, 0, :7] = torch.tensor([3.0, 1.5, 1.5, 6.0, 4.0, 4.0, 0.0], device=device)
    gt_boxes[0, 0, 7] = 2
    gt_boxes[1, 0, :7] = torch.tensor([4.0, 1.5, 1.5, 6.0, 4.0, 4.0, 0.0], device=device)
    gt_boxes[1, 0, 7] = 2

    for name, order in orders.items():
        groups, valid = build_group_tokens_from_mapping(
            order=order,
            flat2win=mappings['flat2win'],
            coords=coords,
            batch_size=batch_size,
            group_size=group_size,
        )
        metrics = summarize_serialized_groups(
            coords=coords,
            group_tokens=groups,
            valid=valid,
            knn_k=4,
            knn_max_queries_per_batch=128,
            boundary_connectivity=26,
            prefix=name,
        )
        metrics.update(summarize_object_fragmentation(
            coords=coords,
            group_tokens=groups,
            valid=valid,
            gt_boxes=gt_boxes,
            prefix=name,
        ))
        assert_finite(metrics, name)
        required = [
            f'{name}/adjacent_dist_mean',
            f'{name}/sjd_p95',
            f'{name}/group_knn_recall',
            f'{name}/boundary_cut_rate',
            f'{name}/object_fragment_groups_mean',
        ]
        missing = [key for key in required if key not in metrics]
        if missing:
            raise AssertionError(f'{name}: missing metrics {missing}')
        if metrics[f'{name}/num_tokens'] != float(coords.shape[0]):
            raise AssertionError(f'{name}: token count mismatch')
        print(
            'serialization_diag_case '
            f'device={device} order={name} adj={metrics[f"{name}/adjacent_dist_mean"]:.4f} '
            f'sjd_p95={metrics[f"{name}/sjd_p95"]:.4f} '
            f'knn={metrics[f"{name}/group_knn_recall"]:.4f} '
            f'cut={metrics[f"{name}/boundary_cut_rate"]:.4f} '
            f'frag={metrics[f"{name}/object_fragment_groups_mean"]:.4f}'
        )


def main():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        run_device(device)
    print('lion_improve_serialization_diagnostics_smoke=pass')


if __name__ == '__main__':
    main()
