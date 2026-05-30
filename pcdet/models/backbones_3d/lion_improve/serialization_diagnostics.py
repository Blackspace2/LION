from typing import Dict, Optional, Sequence, Tuple

import torch


def build_group_tokens_from_mapping(
    order: torch.Tensor,
    flat2win: torch.Tensor,
    coords: torch.Tensor,
    batch_size: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return real LION group tokens plus a mask that excludes padding repeats."""
    if order.dim() != 1:
        raise ValueError(f'order must be 1D, got {tuple(order.shape)}')
    if flat2win.dim() != 1:
        raise ValueError(f'flat2win must be 1D, got {tuple(flat2win.shape)}')
    if coords.dim() != 2 or coords.shape[1] != 4:
        raise ValueError(f'coords must have shape [N,4], got {tuple(coords.shape)}')
    if group_size <= 0:
        raise ValueError(f'group_size must be positive, got {group_size}')
    if flat2win.numel() % group_size != 0:
        raise ValueError(f'flat2win length {flat2win.numel()} is not divisible by group_size={group_size}')
    if order.numel() == 0:
        return (
            torch.empty((0, group_size), device=coords.device, dtype=torch.long),
            torch.empty((0, group_size), device=coords.device, dtype=torch.bool),
        )

    order = order.to(device=coords.device, dtype=torch.long)
    flat2win = flat2win.to(device=coords.device, dtype=torch.long)
    group_tokens = order.index_select(0, flat2win).view(-1, group_size)

    ordered_batch = coords.index_select(0, order)[:, 0].long()
    actual_batch_size = max(int(batch_size), int(coords[:, 0].max().item()) + 1)
    valid_parts = []
    for batch_idx in range(actual_batch_size):
        cur_num = int((ordered_batch == batch_idx).sum().item())
        if cur_num <= 0:
            continue
        padded_num = ((cur_num + group_size - 1) // group_size) * group_size
        pad_num = padded_num - cur_num
        valid_parts.append(torch.cat([
            torch.ones((cur_num,), device=coords.device, dtype=torch.bool),
            torch.zeros((pad_num,), device=coords.device, dtype=torch.bool),
        ]))
    valid = torch.cat(valid_parts, dim=0).view(-1, group_size) if valid_parts else torch.empty(
        (0, group_size), device=coords.device, dtype=torch.bool
    )
    if valid.shape != group_tokens.shape:
        raise RuntimeError(f'valid mask shape {tuple(valid.shape)} does not match groups {tuple(group_tokens.shape)}')
    return group_tokens, valid


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.detach().float().cpu(), q).item())


def _build_group_id(group_tokens: torch.Tensor, valid: torch.Tensor, num_nodes: int) -> torch.Tensor:
    group_id = torch.full((num_nodes,), -1, device=group_tokens.device, dtype=torch.long)
    if group_tokens.numel() == 0:
        return group_id
    group_indices = torch.arange(group_tokens.shape[0], device=group_tokens.device, dtype=torch.long).view(-1, 1)
    group_id[group_tokens[valid]] = group_indices.expand_as(group_tokens)[valid]
    return group_id


def summarize_serialized_groups(
    coords: torch.Tensor,
    group_tokens: torch.Tensor,
    valid: torch.Tensor,
    knn_k: int = 8,
    knn_max_queries_per_batch: int = 1024,
    boundary_connectivity: int = 26,
    prefix: str = '',
) -> Dict[str, float]:
    """Compute locality diagnostics on the exact grouped token stream seen by LION blocks."""
    if coords.dim() != 2 or coords.shape[1] != 4:
        raise ValueError(f'coords must have shape [N,4], got {tuple(coords.shape)}')
    if group_tokens.shape != valid.shape:
        raise ValueError('group_tokens and valid must have the same shape')
    if group_tokens.numel() > 0 and group_tokens.max().item() >= coords.shape[0]:
        raise ValueError('group token index exceeds coord count')

    device = coords.device
    xyz = coords[:, [3, 2, 1]].float()
    metrics: Dict[str, float] = {}
    key_prefix = f'{prefix}/' if prefix else ''

    if group_tokens.shape[1] >= 2 and group_tokens.numel() > 0:
        left = group_tokens[:, :-1]
        right = group_tokens[:, 1:]
        edge_valid = valid[:, :-1] & valid[:, 1:]
        if bool(edge_valid.any()):
            jump = (xyz[left[edge_valid]] - xyz[right[edge_valid]]).norm(dim=1)
        else:
            jump = xyz.new_zeros((0,))
    else:
        jump = xyz.new_zeros((0,))

    metrics[f'{key_prefix}adjacent_dist_mean'] = float(jump.mean().item()) if jump.numel() > 0 else 0.0
    metrics[f'{key_prefix}sjd_mean'] = metrics[f'{key_prefix}adjacent_dist_mean']
    metrics[f'{key_prefix}sjd_p95'] = _quantile(jump, 0.95)
    metrics[f'{key_prefix}sjd_p99'] = _quantile(jump, 0.99)
    metrics[f'{key_prefix}num_jump_edges'] = float(jump.numel())
    metrics[f'{key_prefix}num_groups'] = float(group_tokens.shape[0])
    metrics[f'{key_prefix}num_tokens'] = float(coords.shape[0])

    group_id = _build_group_id(group_tokens, valid, int(coords.shape[0]))
    metrics.update(_summarize_group_knn_recall(
        coords=coords,
        xyz=xyz,
        group_id=group_id,
        knn_k=knn_k,
        max_queries_per_batch=knn_max_queries_per_batch,
        prefix=key_prefix,
    ))
    metrics.update(_summarize_boundary_cut_rate(
        coords=coords,
        group_id=group_id,
        connectivity=boundary_connectivity,
        prefix=key_prefix,
    ))
    return metrics


def _summarize_group_knn_recall(
    coords: torch.Tensor,
    xyz: torch.Tensor,
    group_id: torch.Tensor,
    knn_k: int,
    max_queries_per_batch: int,
    prefix: str,
) -> Dict[str, float]:
    if coords.numel() == 0 or knn_k <= 0:
        return {f'{prefix}group_knn_recall': 0.0, f'{prefix}group_knn_queries': 0.0}
    recalls = []
    query_count = 0
    batches = coords[:, 0].long().unique(sorted=True)
    for batch_idx in batches.tolist():
        batch_indices = (coords[:, 0].long() == int(batch_idx)).nonzero(as_tuple=False).squeeze(1)
        num_batch = int(batch_indices.numel())
        if num_batch <= 1:
            continue
        if num_batch > max_queries_per_batch:
            step = max(num_batch // max_queries_per_batch, 1)
            query_local = torch.arange(0, num_batch, step, device=coords.device, dtype=torch.long)[:max_queries_per_batch]
        else:
            query_local = torch.arange(num_batch, device=coords.device, dtype=torch.long)
        query_indices = batch_indices.index_select(0, query_local)
        dist = torch.cdist(xyz.index_select(0, query_indices), xyz.index_select(0, batch_indices))
        k_eff = min(int(knn_k) + 1, num_batch)
        nearest = torch.topk(dist, k=k_eff, largest=False).indices
        neighbor_global = batch_indices.index_select(0, nearest.reshape(-1)).view(nearest.shape)
        not_self = neighbor_global != query_indices.view(-1, 1)
        neighbor_global = neighbor_global[:, :].masked_select(not_self).view(query_indices.numel(), -1)
        if neighbor_global.numel() == 0:
            continue
        same_group = group_id.index_select(0, neighbor_global.reshape(-1)).view(neighbor_global.shape) == group_id.index_select(
            0, query_indices
        ).view(-1, 1)
        recalls.append(same_group.float().mean(dim=1))
        query_count += int(query_indices.numel())
    if not recalls:
        return {f'{prefix}group_knn_recall': 0.0, f'{prefix}group_knn_queries': 0.0}
    recall = torch.cat(recalls, dim=0)
    return {
        f'{prefix}group_knn_recall': float(recall.mean().item()),
        f'{prefix}group_knn_queries': float(query_count),
    }


def _neighbor_offsets(connectivity: int):
    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                offsets.append((dz, dy, dx))
    return offsets


def _summarize_boundary_cut_rate(
    coords: torch.Tensor,
    group_id: torch.Tensor,
    connectivity: int,
    prefix: str,
) -> Dict[str, float]:
    if connectivity <= 0:
        return {f'{prefix}boundary_cut_rate': 0.0, f'{prefix}boundary_edges': 0.0}
    if coords.numel() == 0:
        return {f'{prefix}boundary_cut_rate': 0.0, f'{prefix}boundary_edges': 0.0}
    coords_cpu = coords.detach().cpu().long()
    group_cpu = group_id.detach().cpu().long()
    coord_to_index = {}
    for idx, row in enumerate(coords_cpu.tolist()):
        coord_to_index.setdefault(tuple(row), idx)

    edge_count = 0
    cut_count = 0
    offsets = _neighbor_offsets(connectivity)
    for idx, row in enumerate(coords_cpu.tolist()):
        batch, z, y, x = row
        for dz, dy, dx in offsets:
            other = (batch, z + dz, y + dy, x + dx)
            other_idx = coord_to_index.get(other)
            if other_idx is None or other_idx <= idx:
                continue
            if group_cpu[idx] < 0 or group_cpu[other_idx] < 0:
                continue
            edge_count += 1
            if int(group_cpu[idx].item()) != int(group_cpu[other_idx].item()):
                cut_count += 1
    return {
        f'{prefix}boundary_cut_rate': float(cut_count) / float(edge_count) if edge_count > 0 else 0.0,
        f'{prefix}boundary_edges': float(edge_count),
    }


def summarize_object_fragmentation(
    coords: torch.Tensor,
    group_tokens: torch.Tensor,
    valid: torch.Tensor,
    gt_boxes: Optional[torch.Tensor],
    voxel_size: Optional[Sequence[float]] = None,
    point_cloud_range: Optional[Sequence[float]] = None,
    stride_xyz: Optional[Sequence[float]] = None,
    prefix: str = '',
) -> Dict[str, float]:
    if gt_boxes is None or coords.numel() == 0:
        return {}
    key_prefix = f'{prefix}/' if prefix else ''
    group_id = _build_group_id(group_tokens, valid, int(coords.shape[0]))
    centers = _voxel_centers_xyz(coords, voxel_size, point_cloud_range, stride_xyz)
    box_indices = _points_in_boxes(coords, centers, gt_boxes)
    if box_indices.numel() == 0:
        return {}

    fragments = []
    voxel_counts = []
    for batch_idx in coords[:, 0].long().unique(sorted=True).tolist():
        batch_mask = coords[:, 0].long() == int(batch_idx)
        cur_box_idx = box_indices[batch_mask]
        cur_global = batch_mask.nonzero(as_tuple=False).squeeze(1)
        for box_idx in cur_box_idx[cur_box_idx >= 0].unique(sorted=True).tolist():
            mask = cur_box_idx == int(box_idx)
            token_idx = cur_global[mask]
            if token_idx.numel() == 0:
                continue
            groups = group_id.index_select(0, token_idx)
            groups = groups[groups >= 0].unique()
            fragments.append(float(groups.numel()))
            voxel_counts.append(float(token_idx.numel()))
    if not fragments:
        return {
            f'{key_prefix}object_count': 0.0,
            f'{key_prefix}object_fragment_groups_mean': 0.0,
            f'{key_prefix}object_voxels_mean': 0.0,
        }
    fragments_t = torch.tensor(fragments, dtype=torch.float32)
    voxels_t = torch.tensor(voxel_counts, dtype=torch.float32)
    return {
        f'{key_prefix}object_count': float(fragments_t.numel()),
        f'{key_prefix}object_fragment_groups_mean': float(fragments_t.mean().item()),
        f'{key_prefix}object_fragment_groups_p95': _quantile(fragments_t, 0.95),
        f'{key_prefix}object_voxels_mean': float(voxels_t.mean().item()),
    }


def _voxel_centers_xyz(
    coords: torch.Tensor,
    voxel_size: Optional[Sequence[float]],
    point_cloud_range: Optional[Sequence[float]],
    stride_xyz: Optional[Sequence[float]],
) -> torch.Tensor:
    xyz = coords[:, [3, 2, 1]].float()
    if voxel_size is None or point_cloud_range is None:
        return xyz
    stride = torch.tensor(stride_xyz or [1.0, 1.0, 1.0], device=coords.device, dtype=torch.float32)
    voxel = torch.tensor(voxel_size, device=coords.device, dtype=torch.float32) * stride
    origin = torch.tensor(point_cloud_range[:3], device=coords.device, dtype=torch.float32)
    return origin.view(1, 3) + (xyz + 0.5) * voxel.view(1, 3)


def _points_in_boxes(coords: torch.Tensor, centers_xyz: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    result = torch.full((coords.shape[0],), -1, device=coords.device, dtype=torch.long)
    if gt_boxes.dim() != 3 or gt_boxes.shape[-1] < 7:
        return result
    try:
        from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

        use_cuda = torch.cuda.is_available()
        points_device = centers_xyz.device if centers_xyz.is_cuda else torch.device('cuda' if use_cuda else 'cpu')
        if points_device.type == 'cuda':
            for batch_idx in coords[:, 0].long().unique(sorted=True).tolist():
                point_mask = coords[:, 0].long() == int(batch_idx)
                cur_points = centers_xyz[point_mask].to(points_device).view(1, -1, 3)
                cur_boxes = gt_boxes[int(batch_idx), :, :7].to(points_device).view(1, -1, 7)
                valid = (cur_boxes[0, :, 3:6] > 0).all(dim=1)
                cur_boxes = cur_boxes[:, valid]
                if cur_boxes.shape[1] == 0 or cur_points.shape[1] == 0:
                    continue
                cur_result = roiaware_pool3d_utils.points_in_boxes_gpu(cur_points, cur_boxes).view(-1)
                result[point_mask] = cur_result.to(result.device)
            return result
    except Exception:
        pass

    # CPU fallback: axis-aligned approximation. Good enough for smoke tests and
    # clearly marked as fallback by only running when CUDA RoI-aware op is absent.
    for batch_idx in coords[:, 0].long().unique(sorted=True).tolist():
        point_mask = coords[:, 0].long() == int(batch_idx)
        points = centers_xyz[point_mask]
        boxes = gt_boxes[int(batch_idx), :, :7].to(device=coords.device, dtype=points.dtype)
        valid = (boxes[:, 3:6] > 0).all(dim=1)
        boxes = boxes[valid]
        if boxes.numel() == 0 or points.numel() == 0:
            continue
        mins = boxes[:, :3] - boxes[:, 3:6] * 0.5
        maxs = boxes[:, :3] + boxes[:, 3:6] * 0.5
        inside = ((points[:, None, :] >= mins[None]) & (points[:, None, :] <= maxs[None])).all(dim=2)
        assigned = torch.where(inside.any(dim=1), inside.float().argmax(dim=1).long(), torch.full(
            (inside.shape[0],), -1, device=coords.device, dtype=torch.long
        ))
        result[point_mask] = assigned
    return result
