import math
from typing import Sequence

import torch

from .topology_order import hilbert_axes_to_distance


GEOMETRY_ORDER_NAMES = {
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
    'hilbert',
}


def _ceil_log2(value: int) -> int:
    value = max(int(value), 1)
    if value <= 1:
        return 1
    return int((value - 1).bit_length())


def _as_int3(values: Sequence[int], name: str):
    values = list(values)
    if len(values) != 3:
        raise ValueError(f'{name} must have 3 values, got {values}')
    return [int(v) for v in values]


def _morton_encode(axes: torch.Tensor, bits: int) -> torch.Tensor:
    if axes.dim() != 2:
        raise ValueError(f'axes must have shape [N,D], got {tuple(axes.shape)}')
    if axes.shape[1] not in (2, 3):
        raise ValueError(f'Morton order currently expects 2 or 3 axes, got D={axes.shape[1]}')
    if bits <= 0:
        return torch.zeros((axes.shape[0],), device=axes.device, dtype=torch.long)
    if axes.shape[1] * bits >= 63:
        raise ValueError(
            f'Morton code requires {axes.shape[1] * bits} bits, which does not fit signed int64'
        )

    axes = axes.long()
    code = torch.zeros((axes.shape[0],), device=axes.device, dtype=torch.long)
    num_dims = int(axes.shape[1])
    for bit in range(int(bits)):
        for axis in range(num_dims):
            code = code | (((axes[:, axis] >> bit) & 1) << (bit * num_dims + axis))
    return code


def _stable_lexsort(keys):
    if len(keys) == 0:
        raise ValueError('lexsort requires at least one key')
    num_nodes = int(keys[0].shape[0])
    device = keys[0].device
    order = torch.arange(num_nodes, device=device, dtype=torch.long)
    for key in reversed(keys):
        key = key.to(device=device)
        if int(key.shape[0]) != num_nodes:
            raise ValueError(f'lexsort key length {int(key.shape[0])} does not match {num_nodes}')
        local = torch.argsort(key.index_select(0, order), stable=True)
        order = order.index_select(0, local)
    return order


def build_geometry_order_from_coords(
    coords: torch.Tensor,
    sparse_shape: Sequence[int],
    window_shape: Sequence[int],
    shift: bool,
    order_name: str,
) -> torch.Tensor:
    """Build a LION window-aware coordinate-only serialization order.

    Coordinates are [batch, z, y, x]. window_shape follows LION's [x, y, z].
    The returned order is a permutation of flat voxel indices and preserves
    LION's batch/window concatenation semantics.
    """
    if coords.dim() != 2 or coords.shape[1] != 4:
        raise ValueError(f'coords must have shape [N,4] in [batch,z,y,x] order, got {tuple(coords.shape)}')
    if order_name not in GEOMETRY_ORDER_NAMES:
        raise ValueError(f'unsupported geometry order: {order_name}')

    num_nodes = int(coords.shape[0])
    device = coords.device
    if num_nodes == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    sparse_z, sparse_y, sparse_x = _as_int3(sparse_shape, 'sparse_shape')
    win_x, win_y, win_z = _as_int3(window_shape, 'window_shape')
    win_x = max(win_x, 1)
    win_y = max(win_y, 1)
    win_z = max(win_z, 1)

    coords = coords.long()
    shift_x = win_x // 2 if shift else 0
    shift_y = win_y // 2 if shift else 0
    shift_z = win_z // 2 if shift else 0

    x = coords[:, 3] + shift_x
    y = coords[:, 2] + shift_y
    z = coords[:, 1] + shift_z

    win_coors_x = torch.div(x, win_x, rounding_mode='floor')
    win_coors_y = torch.div(y, win_y, rounding_mode='floor')
    win_coors_z = torch.div(z, win_z, rounding_mode='floor')

    local_x = x.remainder(win_x)
    local_y = y.remainder(win_y)
    local_z = z.remainder(win_z)

    num_win_x = int(math.ceil(float(sparse_x) / float(win_x)) + 1)
    num_win_y = int(math.ceil(float(sparse_y) / float(win_y)) + 1)
    num_win_z = int(math.ceil(float(sparse_z) / float(win_z)) + 1)
    num_win_per_batch = num_win_x * num_win_y * num_win_z

    batch = coords[:, 0]
    transpose_xy = order_name.endswith('_t')
    if order_name in ('z3d_t', 'h3d_t'):
        transpose_xy = True

    if transpose_xy:
        window_key = (
            batch * num_win_per_batch
            + win_coors_y * num_win_x * num_win_z
            + win_coors_x * num_win_z
            + win_coors_z
        )
        primary_x = local_y
        primary_y = local_x
    else:
        window_key = (
            batch * num_win_per_batch
            + win_coors_x * num_win_y * num_win_z
            + win_coors_y * num_win_z
            + win_coors_z
        )
        primary_x = local_x
        primary_y = local_y

    original_index = torch.arange(num_nodes, device=device, dtype=torch.long)

    if order_name in ('hilbert', 'h3d', 'h3d_t'):
        bits = _ceil_log2(max(win_x, win_y, win_z))
        code = hilbert_axes_to_distance(torch.stack([primary_x, primary_y, local_z], dim=1), bits=bits)
        return _stable_lexsort([window_key, code, original_index])

    if order_name in ('z3d', 'z3d_t'):
        bits = _ceil_log2(max(win_x, win_y, win_z))
        code = _morton_encode(torch.stack([primary_x, primary_y, local_z], dim=1), bits=bits)
        return _stable_lexsort([window_key, code, original_index])

    bits = _ceil_log2(max(win_x, win_y))
    axes_2d = torch.stack([primary_x, primary_y], dim=1)
    if order_name.startswith('bev_h'):
        code = hilbert_axes_to_distance(axes_2d, bits=bits)
    elif order_name.startswith('bev_z'):
        code = _morton_encode(axes_2d, bits=bits)
    else:
        raise ValueError(f'unsupported geometry order: {order_name}')

    if '_25d' in order_name:
        return _stable_lexsort([window_key, code, local_z, original_index])
    return _stable_lexsort([window_key, code, original_index])


def reverse_order_within_batches(order: torch.Tensor, coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    if order.numel() == 0:
        return order
    if coords.dim() != 2 or coords.shape[1] != 4:
        raise ValueError(f'coords must have shape [N,4] in [batch,z,y,x] order, got {tuple(coords.shape)}')
    order = order.to(device=coords.device, dtype=torch.long)
    actual_batch_size = max(int(batch_size), int(coords[:, 0].max().item()) + 1)
    parts = []
    ordered_batch = coords.index_select(0, order)[:, 0].long()
    for batch_idx in range(actual_batch_size):
        mask = ordered_batch == batch_idx
        if bool(mask.any()):
            parts.append(order.index_select(0, mask.nonzero(as_tuple=False).squeeze(1)).flip(0))
    if not parts:
        return order
    return torch.cat(parts, dim=0)
