#!/usr/bin/env python3
import _init_path  # noqa: F401
import torch
from easydict import EasyDict

from pcdet.models.backbones_3d.lion_backbone_one_stride import LION3DBackboneOneStride


SEED = 666


def make_stage_orders(order_pair):
    return EasyDict({
        'ENABLED': True,
        'linear_1': list(order_pair),
        'linear_2': list(order_pair),
        'linear_3': list(order_pair),
        'linear_4': list(order_pair),
        'linear_out': ['x', 'y'],
    })


def make_lion_improve_cfg(order_pair):
    return EasyDict({
        'ENABLED': True,
        'GRAD_LOG_KEYWORDS': ['lion_improve', 'serialization'],
        'GRAPH': EasyDict({
            'ENABLED': True,
            'NEIGHBORHOOD': 26,
            'SIGMA_P': 1.5,
            'SIGMA_RHO': 4.0,
            'RESPONSE_DETACH': True,
            'RESPONSE_GAMMA': 0.25,
            'DENSITY_RANGE_ALPHA': 0.001,
        }),
        'SERIALIZATION': EasyDict({
            'ENABLED': True,
            'EXECUTION_MODE': 'serial',
            'ORDERS': list(order_pair),
            'STAGE_ORDERS': make_stage_orders(order_pair),
            'VALIDATE_MAPPING': True,
            'HEAT_RANK_ITERS': 2,
            'HEAT_RANK_ALPHA': 0.65,
            'RETENTION_TAU': 16.0,
            'TEAR_THRESHOLD': 64.0,
        }),
        'DIAGNOSTICS': EasyDict({'ENABLED': True, 'LOG_INTERVAL': 1, 'SAVE_VIS': False}),
    })


def make_backbone_cfg(order_pair):
    return EasyDict({
        'FEATURE_DIM': 8,
        'NUM_LAYERS': 4,
        'DEPTHS': [1, 1, 1, 1],
        'LAYER_DOWN_SCALES': [
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
        ],
        'WINDOW_SHAPE': [
            [4, 4, 4],
            [4, 4, 2],
            [4, 4, 1],
            [4, 4, 1],
        ],
        'GROUP_SIZE': [16, 16, 16, 16],
        'LAYER_DIM': [8, 8, 8, 8],
        'DIRECTION': ['x', 'y'],
        'DIFF_SCALE': 0.2,
        'DIFFUSION': True,
        'SHIFT': True,
        'OPERATOR': EasyDict({
            'NAME': 'Mamba',
            'CFG': EasyDict({'d_state': 4, 'd_conv': 2, 'expand': 1, 'drop_path': 0.0}),
        }),
        'LION_IMPROVE': make_lion_improve_cfg(order_pair),
    })


def make_batch(device):
    coords = []
    for b in range(2):
        for z in range(0, 16, 2):
            for y in range(0, 5):
                x = (3 * z + 2 * y + b) % 10
                coords.append([b, z, y, x])
                if x + 2 < 10 and y + 1 < 5:
                    coords.append([b, z + 1, y + 1, x + 2])
    coords = torch.tensor(coords, device=device, dtype=torch.int32)
    features = torch.randn((coords.shape[0], 8), device=device, requires_grad=True)
    return {
        'voxel_features': features,
        'voxel_coords': coords,
        'batch_size': 2,
        'global_step': 0,
    }


def grad_norm(model, keyword):
    total = 0.0
    count = 0
    for name, param in model.named_parameters():
        if keyword not in name or param.grad is None:
            continue
        grad = param.grad.detach().float()
        if not torch.isfinite(grad).all():
            raise RuntimeError(f'non-finite gradient: {name}')
        total += float(grad.norm().cpu())
        count += 1
    return total, count


def assert_serialization_flags(model, case_name, topology_expected):
    serial_layers = [
        module for module in model.modules()
        if hasattr(module, 'serialization_execution_mode') and hasattr(module, 'direction')
    ]
    if not serial_layers:
        raise RuntimeError(f'{case_name}: no LIONLayer scan-order modules found')
    for layer in serial_layers:
        if layer.serialization_execution_mode != 'serial':
            raise RuntimeError(f'{case_name}: expected serial execution, got {layer.serialization_execution_mode}')
        if hasattr(layer, 'topology_gate'):
            raise RuntimeError(f'{case_name}: serial scan-order layer should not have fusion gate')
    topology_layers = [layer for layer in serial_layers if getattr(layer, 'topology_order_enabled', False)]
    if topology_expected and len(topology_layers) == 0:
        raise RuntimeError(f'{case_name}: expected at least one SERIALIZATION topology layer')
    if not topology_expected and topology_layers:
        raise RuntimeError(f'{case_name}: geometry-only case unexpectedly enabled SERIALIZATION topology')


def run_case(case_name, order_pair):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LION3DBackboneOneStride(
        make_backbone_cfg(order_pair),
        input_channels=8,
        grid_size=[10, 5, 16],
    ).to(device).train()
    topology_expected = any(order in ('topology', 'topology_rev') for order in order_pair)
    assert_serialization_flags(model, case_name, topology_expected=topology_expected)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = make_batch(device)
    input_norm = batch['voxel_features'].detach().float().norm().clamp_min(1e-6)
    out = model(batch)
    encoded = out['encoded_spconv_tensor']
    if encoded.features.numel() == 0 or not torch.isfinite(encoded.features).all():
        raise RuntimeError(f'{case_name}: invalid encoded features')
    loss = encoded.features.float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f'{case_name}: non-finite loss')
    loss.backward()
    block_grad, block_count = grad_norm(model, 'blocks.')
    if block_count == 0 or block_grad <= 0.0:
        raise RuntimeError(f'{case_name}: expected positive block gradient')
    optimizer.step()

    tb = out.get('lion_improve_tb_dict', {})
    diagnostics_expected = any(order not in ('x', 'y') for order in order_pair)
    if diagnostics_expected and len(tb) == 0:
        raise RuntimeError(f'{case_name}: expected serialization diagnostics')
    if not diagnostics_expected and len(tb) != 0:
        raise RuntimeError(f'{case_name}: baseline axis-order case should not emit serialization diagnostics')

    delta = encoded.features.detach().float().norm() / input_norm
    if float(delta.cpu()) <= 0.0:
        raise RuntimeError(f'{case_name}: expected nonzero output norm')
    peak_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device == 'cuda' else 0.0
    print(
        'serialization_pipeline_case '
        f'name={case_name} orders={list(order_pair)} voxels={int(encoded.features.shape[0])} '
        f'loss={float(loss.detach().cpu()):.8f} block_grad_norm={block_grad:.8e} '
        f'tb_keys={len(tb)} peak_mem_mb={peak_mem:.1f}'
    )


def main():
    cases = [
        ('S0_xy_anchor_path', ['x', 'y']),
        ('S1_bev_z', ['bev_z', 'bev_z_t']),
        ('S2_bev_h', ['bev_h', 'bev_h_t']),
        ('S3_bev_z_25d', ['bev_z_25d', 'bev_z_25d_t']),
        ('S4_bev_h_25d', ['bev_h_25d', 'bev_h_25d_t']),
        ('S5_z3d', ['z3d', 'z3d_t']),
        ('S6_h3d', ['h3d', 'h3d_t']),
        ('S7a_x_topology', ['x', 'topology']),
        ('S7b_y_topology', ['y', 'topology']),
        ('S7c_topology_rev', ['topology', 'topology_rev']),
    ]
    for case_name, order_pair in cases:
        run_case(case_name, order_pair)
    print('lion_improve_serialization_pipeline_smoke=pass')


if __name__ == '__main__':
    main()
