import torch
import torch_scatter


def _compute_scatter_ground_context(is_ground, delta_z, ground_valid, scatter_index, num_bins):
    dtype = delta_z.dtype
    device = delta_z.device

    ones = torch.ones_like(delta_z, dtype=dtype, device=device)
    total_count = torch_scatter.scatter_add(ones, scatter_index, dim=0, dim_size=num_bins)
    ground_count = torch_scatter.scatter_add(is_ground.to(dtype), scatter_index, dim=0, dim_size=num_bins)
    valid_count = torch_scatter.scatter_add(ground_valid.to(dtype), scatter_index, dim=0, dim_size=num_bins)

    valid_delta = delta_z * ground_valid.to(dtype)
    valid_delta_sq = delta_z.square() * ground_valid.to(dtype)
    delta_sum = torch_scatter.scatter_add(valid_delta, scatter_index, dim=0, dim_size=num_bins)
    delta_sq_sum = torch_scatter.scatter_add(valid_delta_sq, scatter_index, dim=0, dim_size=num_bins)

    mean_delta = torch.zeros(num_bins, dtype=dtype, device=device)
    std_delta = torch.zeros(num_bins, dtype=dtype, device=device)
    valid_ratio = torch.zeros(num_bins, dtype=dtype, device=device)
    ground_ratio = torch.zeros(num_bins, dtype=dtype, device=device)

    has_valid = valid_count > 0
    mean_delta[has_valid] = delta_sum[has_valid] / valid_count[has_valid]
    var_delta = torch.zeros_like(mean_delta)
    var_delta[has_valid] = delta_sq_sum[has_valid] / valid_count[has_valid] - mean_delta[has_valid].square()
    std_delta[has_valid] = torch.sqrt(var_delta[has_valid].clamp_min(0.0))

    has_total = total_count > 0
    valid_ratio[has_total] = valid_count[has_total] / total_count[has_total]
    ground_ratio[has_total] = ground_count[has_total] / total_count[has_total]

    return torch.stack([mean_delta, std_delta, ground_ratio, valid_ratio], dim=1)


def build_voxel_ground_context_from_inv(points, unq_inv, num_voxels, feature_indices):
    if points.numel() == 0 or num_voxels == 0:
        return points.new_zeros((num_voxels, 4))

    is_ground = points[:, feature_indices['is_ground']]
    delta_z = points[:, feature_indices['delta_z_to_ground']]
    ground_valid = points[:, feature_indices['ground_valid']]
    return _compute_scatter_ground_context(
        is_ground=is_ground,
        delta_z=delta_z,
        ground_valid=ground_valid,
        scatter_index=unq_inv,
        num_bins=num_voxels
    )


def pool_ground_context_to_sparse_coords(
    points,
    target_coords,
    target_spatial_shape,
    voxel_size,
    point_cloud_range,
    stride_xyz,
    feature_indices
):
    num_targets = int(target_coords.shape[0])
    if num_targets == 0:
        return target_coords.new_zeros((0, 4)).to(dtype=points.dtype)
    if points.numel() == 0:
        return points.new_zeros((num_targets, 4))

    stride_xyz = torch.as_tensor(stride_xyz, device=points.device, dtype=points.dtype).view(1, 3)
    voxel_size = voxel_size.view(1, 3).to(device=points.device, dtype=points.dtype)
    point_cloud_range = point_cloud_range.view(-1).to(device=points.device, dtype=points.dtype)
    spatial_shape_zyx = torch.as_tensor(target_spatial_shape, device=points.device, dtype=torch.long)
    spatial_shape_xyz = torch.stack(
        [spatial_shape_zyx[2], spatial_shape_zyx[1], spatial_shape_zyx[0]], dim=0
    )

    point_xyz = points[:, 1:4]
    point_coords_xyz = torch.floor((point_xyz - point_cloud_range[:3].view(1, 3)) / (voxel_size * stride_xyz)).long()
    valid_mask = (
        (point_coords_xyz >= 0) &
        (point_coords_xyz < spatial_shape_xyz.view(1, 3))
    ).all(dim=1)
    if not bool(valid_mask.any()):
        return points.new_zeros((num_targets, 4))

    points = points[valid_mask]
    point_coords_xyz = point_coords_xyz[valid_mask]

    scale_xyz = int(spatial_shape_xyz[0].item() * spatial_shape_xyz[1].item() * spatial_shape_xyz[2].item())
    scale_yz = int(spatial_shape_xyz[1].item() * spatial_shape_xyz[2].item())
    scale_z = int(spatial_shape_xyz[2].item())

    point_merge_coords = (
        points[:, 0].long() * scale_xyz +
        point_coords_xyz[:, 0] * scale_yz +
        point_coords_xyz[:, 1] * scale_z +
        point_coords_xyz[:, 2]
    )
    unq_merge, unq_inv = torch.unique(point_merge_coords, return_inverse=True, dim=0)
    pooled_context = build_voxel_ground_context_from_inv(
        points=points,
        unq_inv=unq_inv,
        num_voxels=int(unq_merge.shape[0]),
        feature_indices=feature_indices
    )

    target_merge_coords = (
        target_coords[:, 0].long() * scale_xyz +
        target_coords[:, 3].long() * scale_yz +
        target_coords[:, 2].long() * scale_z +
        target_coords[:, 1].long()
    )
    sorted_merge, sorted_order = torch.sort(unq_merge)
    target_positions = torch.bucketize(target_merge_coords, sorted_merge)
    matched = (
        (target_positions < sorted_merge.shape[0]) &
        (sorted_merge[target_positions.clamp(max=max(sorted_merge.shape[0] - 1, 0))] == target_merge_coords)
    )

    gathered_context = points.new_zeros((num_targets, 4))
    if bool(matched.any()):
        matched_positions = target_positions[matched]
        matched_indices = sorted_order[matched_positions]
        gathered_context[matched] = pooled_context[matched_indices]
    return gathered_context
