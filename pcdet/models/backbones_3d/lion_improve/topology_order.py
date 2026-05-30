from typing import Optional, Sequence

import torch

from .topology_context import SerializationGraphContext, cfg_get


def _normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    return (values - min_value) / (max_value - min_value).clamp_min(1e-6)


def _diffuse_score(context: SerializationGraphContext, score: torch.Tensor, num_iters: int, alpha: float) -> torch.Tensor:
    if context.num_edges == 0 or num_iters <= 0:
        return score

    src, dst = context.edge_index[0], context.edge_index[1]
    degree = context.degree.clamp_min(1e-6)
    alpha = min(max(float(alpha), 0.0), 1.0)
    for _ in range(int(num_iters)):
        neighbor_sum = torch.zeros_like(score)
        neighbor_sum.scatter_add_(0, src, context.edge_weight.to(score.dtype) * score[dst])
        neighbor_mean = neighbor_sum / degree.to(score.dtype)
        score = (1.0 - alpha) * score + alpha * neighbor_mean
    return score


def _infer_hilbert_bits(spatial_shape: Sequence[int]) -> int:
    shape = torch.as_tensor(spatial_shape, dtype=torch.long).view(-1)
    if shape.numel() != 3:
        raise ValueError(f'spatial_shape must have 3 values [z,y,x], got {spatial_shape}')
    max_dim = int(shape.clamp_min(1).max().item())
    if max_dim <= 1:
        return 1
    return int((max_dim - 1).bit_length())


def hilbert_axes_to_distance(axes: torch.Tensor, bits: int) -> torch.Tensor:
    if axes.dim() != 2:
        raise ValueError(f'axes must have shape [N,D], got {tuple(axes.shape)}')
    if axes.shape[1] not in (2, 3):
        raise ValueError(f'Hilbert order currently expects 2 or 3 axes, got D={axes.shape[1]}')
    if bits <= 0:
        return torch.zeros((axes.shape[0],), device=axes.device, dtype=torch.long)
    if axes.numel() == 0:
        return torch.empty((0,), device=axes.device, dtype=torch.long)
    if axes.shape[1] * bits >= 63:
        raise ValueError(
            f'Hilbert distance requires {axes.shape[1] * bits} bits, which does not fit signed int64'
        )

    x = axes.long().clone()
    num_dims = int(x.shape[1])
    q = 1 << (bits - 1)
    while q > 1:
        lower_mask = q - 1
        for axis in range(num_dims):
            high_bit_set = (x[:, axis] & q) != 0
            x0_if_set = x[:, 0] ^ lower_mask
            swap_mask = (x[:, 0] ^ x[:, axis]) & lower_mask
            x0_if_clear = x[:, 0] ^ swap_mask
            xa_if_clear = x[:, axis] ^ swap_mask
            x[:, 0] = torch.where(high_bit_set, x0_if_set, x0_if_clear)
            x[:, axis] = torch.where(high_bit_set, x[:, axis], xa_if_clear)
        q >>= 1

    for axis in range(1, num_dims):
        x[:, axis] ^= x[:, axis - 1]

    t = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
    q = 1 << (bits - 1)
    while q > 1:
        t = torch.where((x[:, num_dims - 1] & q) != 0, t ^ (q - 1), t)
        q >>= 1
    x ^= t.view(-1, 1)

    distance = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
    for bit in range(bits - 1, -1, -1):
        for axis in range(num_dims):
            distance = distance * 2 + ((x[:, axis] >> bit) & 1)
    return distance


def build_hilbert_order_from_coords(
    coords: torch.Tensor,
    spatial_shape: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    if coords.dim() != 2 or coords.shape[1] != 4:
        raise ValueError(f'coords must have shape [N,4] in [batch,z,y,x] order, got {tuple(coords.shape)}')
    num_nodes = int(coords.shape[0])
    device = coords.device
    if num_nodes == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    coords = coords.long()
    spatial_shape_cpu = torch.as_tensor(spatial_shape, dtype=torch.long).view(-1)
    if spatial_shape_cpu.numel() != 3:
        raise ValueError(f'spatial_shape must have 3 values [z,y,x], got {spatial_shape}')

    bits = _infer_hilbert_bits(spatial_shape_cpu.tolist())
    axes_xyz = coords[:, [3, 2, 1]]
    hilbert_key = hilbert_axes_to_distance(axes_xyz, bits=bits)

    order_parts = []
    actual_batch_size = max(int(batch_size), int(coords[:, 0].max().item()) + 1)
    for batch_idx in range(actual_batch_size):
        batch_mask = coords[:, 0] == batch_idx
        if not bool(batch_mask.any()):
            continue
        batch_indices = batch_mask.nonzero(as_tuple=False).squeeze(1)
        local_key = hilbert_key.index_select(0, batch_indices)
        local_order = torch.argsort(local_key, stable=True)
        order_parts.append(batch_indices.index_select(0, local_order))

    if not order_parts:
        return torch.arange(num_nodes, device=device, dtype=torch.long)
    order = torch.cat(order_parts, dim=0)
    if order.numel() != num_nodes:
        raise RuntimeError(f'Hilbert order length {order.numel()} does not match node count {num_nodes}')
    return order


def build_topology_order_from_context(
    context: SerializationGraphContext,
    fallback_order: Optional[torch.Tensor] = None,
    cfg=None,
) -> torch.Tensor:
    num_nodes = context.num_nodes
    device = context.coords.device
    if num_nodes == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    if fallback_order is None:
        fallback_order = torch.arange(num_nodes, device=device, dtype=torch.long)
    else:
        fallback_order = fallback_order.to(device=device, dtype=torch.long)
        if fallback_order.numel() != num_nodes:
            raise ValueError(f'fallback_order length {fallback_order.numel()} does not match {num_nodes}')

    if num_nodes <= 2 or context.num_edges == 0:
        return fallback_order

    serialization_cfg = cfg_get(cfg, 'SERIALIZATION', cfg)
    num_iters = int(cfg_get(serialization_cfg, 'HEAT_RANK_ITERS', 4))
    alpha = float(cfg_get(serialization_cfg, 'HEAT_RANK_ALPHA', 0.65))
    density_weight = float(cfg_get(serialization_cfg, 'DENSITY_TIE_WEIGHT', 0.05))
    response_weight = float(cfg_get(serialization_cfg, 'RESPONSE_TIE_WEIGHT', -0.03))

    xyz = context.positions_xyz
    xyz_norm = torch.stack([_normalize(xyz[:, i]) for i in range(3)], dim=1)
    density = _normalize(context.density.to(dtype=xyz.dtype))
    response = _normalize(context.response.to(dtype=xyz.dtype))

    # A deterministic heat coordinate: start from a spatial Morton-like scalar,
    # then smooth it over the metric-measure graph so strong local edges stay close.
    score = (
        xyz_norm[:, 0]
        + 0.37 * xyz_norm[:, 1]
        + 0.13 * xyz_norm[:, 2]
        + density_weight * density
        + response_weight * response
    )
    score = _diffuse_score(context, score, num_iters=num_iters, alpha=alpha)

    inv_fallback = torch.empty_like(fallback_order)
    inv_fallback[fallback_order] = torch.arange(num_nodes, device=device, dtype=torch.long)
    fallback_tie = inv_fallback.to(dtype=score.dtype) / max(num_nodes - 1, 1)
    score = score + fallback_tie * 1e-4

    order_parts = []
    actual_batch_size = max(context.batch_size, int(context.coords[:, 0].max().item()) + 1)
    for batch_idx in range(actual_batch_size):
        batch_mask = context.coords[:, 0] == batch_idx
        if not bool(batch_mask.any()):
            continue
        batch_indices = batch_mask.nonzero(as_tuple=False).squeeze(1)
        local_score = score.index_select(0, batch_indices)
        local_order = torch.argsort(local_score, stable=True)
        order_parts.append(batch_indices.index_select(0, local_order))

    if not order_parts:
        return fallback_order
    order = torch.cat(order_parts, dim=0)
    if order.numel() != num_nodes:
        raise RuntimeError(f'topology order length {order.numel()} does not match node count {num_nodes}')
    return order
