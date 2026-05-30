from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class SerializationGraphContext:
    coords: torch.Tensor
    spatial_shape: torch.Tensor
    batch_size: int
    positions_xyz: torch.Tensor
    density: torch.Tensor
    response: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    degree: torch.Tensor
    neighbor_count: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def _tensor_1d(values: Optional[Sequence[float]], device, dtype, length: int, fill: float) -> torch.Tensor:
    if values is None:
        return torch.full((length,), fill, device=device, dtype=dtype)
    tensor = torch.as_tensor(values, device=device, dtype=dtype).view(-1)
    if tensor.numel() != length:
        raise ValueError(f'Expected {length} values, got {tensor.numel()}: {values}')
    return tensor


def _spatial_shape_tensor(coords: torch.Tensor, spatial_shape: Optional[Sequence[int]]) -> torch.Tensor:
    if spatial_shape is not None:
        shape = torch.as_tensor(spatial_shape, device=coords.device, dtype=torch.long).view(-1)
        if shape.numel() != 3:
            raise ValueError(f'spatial_shape must have 3 values [z,y,x], got {spatial_shape}')
        return shape
    if coords.numel() == 0:
        return torch.zeros((3,), device=coords.device, dtype=torch.long)
    return coords[:, 1:4].max(dim=0).values.long() + 1


def _infer_batch_size(coords: torch.Tensor, batch_size: Optional[int]) -> int:
    if batch_size is not None:
        return int(batch_size)
    if coords.numel() == 0:
        return 0
    return int(coords[:, 0].max().item()) + 1


def _encode_coords(coords: torch.Tensor, spatial_shape: torch.Tensor) -> torch.Tensor:
    z_size = spatial_shape[0].clamp_min(1).long()
    y_size = spatial_shape[1].clamp_min(1).long()
    x_size = spatial_shape[2].clamp_min(1).long()
    batch_stride = z_size * y_size * x_size
    z_stride = y_size * x_size
    y_stride = x_size
    return (
        coords[:, 0].long() * batch_stride
        + coords[:, 1].long() * z_stride
        + coords[:, 2].long() * y_stride
        + coords[:, 3].long()
    )


def _neighbor_offsets(neighborhood: int, device: torch.device) -> torch.Tensor:
    if neighborhood not in (6, 18, 26):
        raise ValueError(f'LION_IMPROVE.GRAPH.NEIGHBORHOOD expects 6, 18, or 26, got {neighborhood}')

    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                manhattan = abs(dz) + abs(dy) + abs(dx)
                if neighborhood == 6 and manhattan != 1:
                    continue
                if neighborhood == 18 and manhattan > 2:
                    continue
                offsets.append((dz, dy, dx))
    return torch.tensor(offsets, device=device, dtype=torch.long)


def _build_neighbor_edges(coords: torch.Tensor, spatial_shape: torch.Tensor, neighborhood: int) -> torch.Tensor:
    num_nodes = int(coords.shape[0])
    if num_nodes <= 1:
        return torch.empty((2, 0), device=coords.device, dtype=torch.long)

    sorted_codes, sorted_order = torch.sort(_encode_coords(coords, spatial_shape))
    shape = spatial_shape.view(1, 3)
    src_parts = []
    dst_parts = []
    offsets = _neighbor_offsets(neighborhood, coords.device)
    base_zyx = coords[:, 1:4].long()

    for offset in offsets:
        nbr_zyx = base_zyx + offset.view(1, 3)
        valid = ((nbr_zyx >= 0) & (nbr_zyx < shape)).all(dim=1)
        if not bool(valid.any()):
            continue

        target_coords = torch.cat([coords[:, 0:1].long(), nbr_zyx], dim=1)
        target_codes = _encode_coords(target_coords[valid], spatial_shape)
        pos = torch.searchsorted(sorted_codes, target_codes)
        in_range = pos < sorted_codes.shape[0]
        if not bool(in_range.any()):
            continue

        src = valid.nonzero(as_tuple=False).squeeze(1)[in_range]
        pos = pos[in_range]
        matched = sorted_codes[pos] == target_codes[in_range]
        if not bool(matched.any()):
            continue
        src_parts.append(src[matched])
        dst_parts.append(sorted_order[pos[matched]])

    duplicate = sorted_codes[1:] == sorted_codes[:-1]
    if bool(duplicate.any()):
        left = sorted_order[:-1][duplicate]
        right = sorted_order[1:][duplicate]
        src_parts.extend([left, right])
        dst_parts.extend([right, left])

    if not src_parts:
        return torch.empty((2, 0), device=coords.device, dtype=torch.long)
    return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)


def _compute_positions_xyz(
    coords: torch.Tensor,
    voxel_size: Optional[Sequence[float]],
    point_cloud_range: Optional[Sequence[float]],
    stride_xyz: Optional[Sequence[float]],
    dtype: torch.dtype,
) -> torch.Tensor:
    xyz_coords = coords[:, [3, 2, 1]].to(dtype=dtype)
    voxel_size_t = _tensor_1d(voxel_size, coords.device, dtype, 3, 1.0)
    stride_t = _tensor_1d(stride_xyz, coords.device, dtype, 3, 1.0)
    if point_cloud_range is None:
        pc_min = torch.zeros((3,), device=coords.device, dtype=dtype)
    else:
        pc_min = torch.as_tensor(point_cloud_range, device=coords.device, dtype=dtype).view(-1)[:3]
        if pc_min.numel() != 3:
            raise ValueError(f'point_cloud_range must have at least 3 values, got {point_cloud_range}')
    return pc_min.view(1, 3) + (xyz_coords + 0.5) * (voxel_size_t * stride_t).view(1, 3)


def _compute_density(
    voxel_num_points: Optional[torch.Tensor],
    positions_xyz: torch.Tensor,
    dtype: torch.dtype,
    range_alpha: float,
) -> torch.Tensor:
    num_nodes = int(positions_xyz.shape[0])
    if voxel_num_points is None:
        counts = torch.ones((num_nodes,), device=positions_xyz.device, dtype=dtype)
    else:
        counts = voxel_num_points.to(device=positions_xyz.device, dtype=dtype).view(-1)
        if counts.numel() != num_nodes:
            raise ValueError(f'voxel_num_points length {counts.numel()} does not match coords length {num_nodes}')
        counts = counts.clamp_min(0.0)
    ranges = positions_xyz.norm(dim=1)
    return torch.log1p(counts) * (1.0 + float(range_alpha) * ranges.square())


def _compute_response(
    features: Optional[torch.Tensor],
    response: Optional[torch.Tensor],
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    detach: bool,
) -> torch.Tensor:
    if response is not None:
        resp = response.to(device=device, dtype=dtype).view(-1)
        if resp.numel() != num_nodes:
            raise ValueError(f'response length {resp.numel()} does not match coords length {num_nodes}')
    elif features is not None and features.numel() > 0:
        resp = features.detach() if detach else features
        resp = resp.to(dtype=dtype).float().norm(dim=1).to(dtype=dtype)
    else:
        resp = torch.zeros((num_nodes,), device=device, dtype=dtype)

    if resp.numel() == 0:
        return resp
    resp_min = resp.min()
    resp_max = resp.max()
    return (resp - resp_min) / (resp_max - resp_min).clamp_min(1e-6)


def build_serialization_graph_context(
    coords: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    voxel_num_points: Optional[torch.Tensor] = None,
    spatial_shape: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    voxel_size: Optional[Sequence[float]] = None,
    point_cloud_range: Optional[Sequence[float]] = None,
    stride_xyz: Optional[Sequence[float]] = None,
    cfg: Any = None,
    response: Optional[torch.Tensor] = None,
) -> SerializationGraphContext:
    if coords.dim() != 2 or coords.shape[1] != 4:
        raise ValueError(f'coords must have shape [N,4] in [batch,z,y,x] order, got {tuple(coords.shape)}')

    coords = coords.long()
    dtype = features.dtype if features is not None and torch.is_floating_point(features) else torch.float32
    spatial_shape_t = _spatial_shape_tensor(coords, spatial_shape)
    batch_size_i = _infer_batch_size(coords, batch_size)
    num_nodes = int(coords.shape[0])

    if num_nodes == 0:
        empty_float = torch.empty((0,), device=coords.device, dtype=dtype)
        return SerializationGraphContext(
            coords=coords,
            spatial_shape=spatial_shape_t,
            batch_size=batch_size_i,
            positions_xyz=torch.empty((0, 3), device=coords.device, dtype=dtype),
            density=empty_float,
            response=empty_float,
            edge_index=torch.empty((2, 0), device=coords.device, dtype=torch.long),
            edge_weight=empty_float,
            degree=empty_float,
            neighbor_count=empty_float,
        )

    neighborhood = int(cfg_get(cfg_get(cfg, 'GRAPH', cfg), 'NEIGHBORHOOD', cfg_get(cfg, 'NEIGHBORHOOD', 26)))
    sigma_p = float(cfg_get(cfg_get(cfg, 'GRAPH', cfg), 'SIGMA_P', cfg_get(cfg, 'SIGMA_P', 1.5)))
    sigma_rho = float(cfg_get(cfg_get(cfg, 'GRAPH', cfg), 'SIGMA_RHO', cfg_get(cfg, 'SIGMA_RHO', 1.0)))
    response_detach = bool(cfg_get(cfg_get(cfg, 'GRAPH', cfg), 'RESPONSE_DETACH', cfg_get(cfg, 'RESPONSE_DETACH', True)))
    response_gamma = float(cfg_get(cfg_get(cfg, 'GRAPH', cfg), 'RESPONSE_GAMMA', cfg_get(cfg, 'RESPONSE_GAMMA', 0.25)))
    range_alpha = float(cfg_get(cfg_get(cfg, 'GRAPH', cfg), 'DENSITY_RANGE_ALPHA', cfg_get(cfg, 'DENSITY_RANGE_ALPHA', 0.001)))

    positions_xyz = _compute_positions_xyz(coords, voxel_size, point_cloud_range, stride_xyz, dtype)
    density = _compute_density(voxel_num_points, positions_xyz, dtype, range_alpha)
    response_t = _compute_response(features, response, num_nodes, coords.device, dtype, response_detach)
    edge_index = _build_neighbor_edges(coords, spatial_shape_t, neighborhood)

    if edge_index.shape[1] == 0:
        empty_float = torch.empty((0,), device=coords.device, dtype=dtype)
        return SerializationGraphContext(
            coords=coords,
            spatial_shape=spatial_shape_t,
            batch_size=batch_size_i,
            positions_xyz=positions_xyz,
            density=density,
            response=response_t,
            edge_index=edge_index,
            edge_weight=empty_float,
            degree=torch.zeros((num_nodes,), device=coords.device, dtype=dtype),
            neighbor_count=torch.zeros((num_nodes,), device=coords.device, dtype=dtype),
        )

    src, dst = edge_index[0], edge_index[1]
    delta = positions_xyz[src] - positions_xyz[dst]
    dist2 = delta.square().sum(dim=1)
    rho_delta = (density[src] - density[dst]).abs()
    sigma_p_sq = max(sigma_p * sigma_p, 1e-6)
    sigma_rho = max(sigma_rho, 1e-6)
    edge_weight = torch.exp(-dist2 / sigma_p_sq) * torch.exp(-rho_delta / sigma_rho)
    edge_weight = edge_weight * (1.0 + response_gamma * response_t[src] * response_t[dst])
    edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=dtype)

    degree = torch.zeros((num_nodes,), device=coords.device, dtype=dtype)
    degree.scatter_add_(0, src, edge_weight)
    neighbor_count = torch.zeros((num_nodes,), device=coords.device, dtype=dtype)
    neighbor_count.scatter_add_(0, src, torch.ones_like(edge_weight))

    return SerializationGraphContext(
        coords=coords,
        spatial_shape=spatial_shape_t,
        batch_size=batch_size_i,
        positions_xyz=positions_xyz,
        density=density,
        response=response_t,
        edge_index=edge_index,
        edge_weight=edge_weight,
        degree=degree,
        neighbor_count=neighbor_count,
    )
