from typing import Dict

import torch

from .topology_context import SerializationGraphContext


def _scalar(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.detach().float().mean().item())
    return float(value)


def _finite_ratio(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 1.0
    return float(torch.isfinite(value).float().mean().item())


def diagnose_graph_context(context: SerializationGraphContext, prefix: str = 'lion_improve/graph') -> Dict[str, float]:
    return {
        f'{prefix}/num_nodes': float(context.num_nodes),
        f'{prefix}/num_edges': float(context.num_edges),
        f'{prefix}/degree_mean': _scalar(context.degree),
        f'{prefix}/degree_max': _scalar(context.degree.max() if context.degree.numel() > 0 else None),
        f'{prefix}/neighbor_count_mean': _scalar(context.neighbor_count),
        f'{prefix}/density_mean': _scalar(context.density),
        f'{prefix}/response_mean': _scalar(context.response),
        f'{prefix}/edge_weight_mean': _scalar(context.edge_weight),
        f'{prefix}/edge_weight_finite_ratio': _finite_ratio(context.edge_weight),
    }


def diagnose_order(
    context: SerializationGraphContext,
    order: torch.Tensor,
    prefix: str,
    retention_tau: float = 16.0,
    tear_threshold: float = 64.0,
) -> Dict[str, float]:
    if context.num_nodes == 0:
        return {
            f'{prefix}/info_retention': 1.0,
            f'{prefix}/weighted_tear_rate': 0.0,
            f'{prefix}/successor_distance_mean': 0.0,
            f'{prefix}/successor_distance_max': 0.0,
        }
    if order.numel() != context.num_nodes:
        raise ValueError(f'order length {order.numel()} does not match context nodes {context.num_nodes}')

    order = order.to(device=context.coords.device, dtype=torch.long)
    if order.numel() > 0:
        sorted_order = torch.sort(order).values
        expected = torch.arange(context.num_nodes, device=order.device, dtype=torch.long)
        if not torch.equal(sorted_order, expected):
            raise ValueError('order must be a permutation of node indices')

    inv_pos = torch.empty_like(order)
    inv_pos[order] = torch.arange(context.num_nodes, device=order.device, dtype=torch.long)

    if context.num_edges == 0:
        info_retention = torch.ones((), device=order.device, dtype=context.positions_xyz.dtype)
        tear_rate = torch.zeros((), device=order.device, dtype=context.positions_xyz.dtype)
    else:
        src, dst = context.edge_index[0], context.edge_index[1]
        rank_dist = (inv_pos[src] - inv_pos[dst]).abs().to(dtype=context.edge_weight.dtype)
        weights = context.edge_weight
        denom = weights.sum().clamp_min(1e-6)
        kernel = torch.exp(-rank_dist / max(float(retention_tau), 1e-6))
        info_retention = (weights * kernel).sum() / denom
        tear_rate = (weights * (rank_dist > float(tear_threshold)).to(weights.dtype)).sum() / denom

    if context.num_nodes <= 1:
        succ_mean = torch.zeros((), device=order.device, dtype=context.positions_xyz.dtype)
        succ_max = succ_mean
    else:
        ordered_pos = context.positions_xyz[order]
        succ_dist = (ordered_pos[1:] - ordered_pos[:-1]).norm(dim=1)
        succ_mean = succ_dist.mean()
        succ_max = succ_dist.max()

    return {
        f'{prefix}/info_retention': float(info_retention.detach().item()),
        f'{prefix}/weighted_tear_rate': float(tear_rate.detach().item()),
        f'{prefix}/successor_distance_mean': float(succ_mean.detach().item()),
        f'{prefix}/successor_distance_max': float(succ_max.detach().item()),
    }

