import torch
import torch_scatter


PATCH_CONTEXT_RAW_DIM = 10


def build_voxel_patch_context_from_inv(points, unq_inv, num_voxels, feature_indices):
    if points.numel() == 0 or num_voxels == 0:
        empty_context = points.new_zeros((num_voxels, PATCH_CONTEXT_RAW_DIM))
        empty_patch_ids = torch.full((num_voxels,), -1, dtype=torch.long, device=points.device)
        return empty_context, empty_patch_ids

    dtype = points.dtype
    device = points.device

    is_ground = points[:, feature_indices['is_ground']]
    patch_center_z = points[:, feature_indices['patch_center_z']]
    patch_normal_z = points[:, feature_indices['patch_normal_z']]
    patch_flatness = points[:, feature_indices['patch_flatness']]
    patch_elevation = points[:, feature_indices['patch_elevation']]
    patch_point_count = points[:, feature_indices['patch_point_count']]
    patch_zone_id = points[:, feature_indices['patch_zone_id']]
    patch_ring_id = points[:, feature_indices['patch_ring_id']]
    patch_sector_id = points[:, feature_indices['patch_sector_id']]
    point_patch_id = torch.round(points[:, feature_indices['point_patch_id']]).long()

    ones = torch.ones_like(is_ground, dtype=dtype, device=device)
    total_count = torch_scatter.scatter_add(ones, unq_inv, dim=0, dim_size=num_voxels)
    ground_count = torch_scatter.scatter_add(is_ground.to(dtype), unq_inv, dim=0, dim_size=num_voxels)

    context = points.new_zeros((num_voxels, PATCH_CONTEXT_RAW_DIM))
    voxel_patch_ids = torch.full((num_voxels,), -1, dtype=torch.long, device=device)
    has_points = total_count > 0
    context[has_points, 0] = ground_count[has_points] / total_count[has_points].clamp_min(1.0)

    valid_patch = point_patch_id >= 0
    if not bool(valid_patch.any()):
        return context, voxel_patch_ids

    max_patch_id = int(point_patch_id[valid_patch].max().item()) + 2
    encoded_patch_id = torch.where(valid_patch, point_patch_id + 1, point_patch_id.new_zeros(point_patch_id.shape))
    pair_key = unq_inv.long() * max_patch_id + encoded_patch_id
    uniq_pair, pair_inv, pair_counts = torch.unique(pair_key, return_inverse=True, return_counts=True)
    pair_voxel = uniq_pair // max_patch_id
    pair_patch_id = uniq_pair % max_patch_id - 1

    pair_center_z = torch_scatter.scatter_mean(patch_center_z, pair_inv, dim=0)
    pair_normal_z = torch_scatter.scatter_mean(patch_normal_z, pair_inv, dim=0)
    pair_flatness = torch_scatter.scatter_mean(patch_flatness, pair_inv, dim=0)
    pair_elevation = torch_scatter.scatter_mean(patch_elevation, pair_inv, dim=0)
    pair_point_count = torch_scatter.scatter_mean(patch_point_count, pair_inv, dim=0)
    pair_zone_id = torch_scatter.scatter_mean(patch_zone_id, pair_inv, dim=0)
    pair_ring_id = torch_scatter.scatter_mean(patch_ring_id, pair_inv, dim=0)
    pair_sector_id = torch_scatter.scatter_mean(patch_sector_id, pair_inv, dim=0)

    dominant_count, dominant_pair_idx = torch_scatter.scatter_max(
        pair_counts.to(dtype), pair_voxel, dim=0, dim_size=num_voxels
    )
    valid_dominant = dominant_pair_idx >= 0
    if not bool(valid_dominant.any()):
        return context, voxel_patch_ids

    selected_pair_idx = dominant_pair_idx[valid_dominant]
    voxel_patch_ids[valid_dominant] = pair_patch_id[selected_pair_idx]
    context[valid_dominant, 1] = pair_center_z[selected_pair_idx]
    context[valid_dominant, 2] = pair_normal_z[selected_pair_idx]
    context[valid_dominant, 3] = pair_flatness[selected_pair_idx]
    context[valid_dominant, 4] = pair_elevation[selected_pair_idx]
    context[valid_dominant, 5] = pair_point_count[selected_pair_idx]
    context[valid_dominant, 6] = pair_zone_id[selected_pair_idx]
    context[valid_dominant, 7] = pair_ring_id[selected_pair_idx]
    context[valid_dominant, 8] = pair_sector_id[selected_pair_idx]
    context[valid_dominant, 9] = dominant_count[valid_dominant] / total_count[valid_dominant].clamp_min(1.0)
    return context, voxel_patch_ids


def pool_patch_context_to_sparse_coords(
    points,
    target_coords,
    target_spatial_shape,
    voxel_size,
    point_cloud_range,
    stride_xyz,
    feature_indices,
):
    num_targets = int(target_coords.shape[0])
    if num_targets == 0:
        empty_context = target_coords.new_zeros((0, PATCH_CONTEXT_RAW_DIM)).to(dtype=points.dtype)
        empty_patch_ids = torch.full((0,), -1, dtype=torch.long, device=target_coords.device)
        return empty_context, empty_patch_ids
    if points.numel() == 0:
        empty_context = points.new_zeros((num_targets, PATCH_CONTEXT_RAW_DIM))
        empty_patch_ids = torch.full((num_targets,), -1, dtype=torch.long, device=points.device)
        return empty_context, empty_patch_ids

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
        empty_context = points.new_zeros((num_targets, PATCH_CONTEXT_RAW_DIM))
        empty_patch_ids = torch.full((num_targets,), -1, dtype=torch.long, device=points.device)
        return empty_context, empty_patch_ids

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
    pooled_context, pooled_patch_ids = build_voxel_patch_context_from_inv(
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

    gathered_context = points.new_zeros((num_targets, PATCH_CONTEXT_RAW_DIM))
    gathered_patch_ids = torch.full((num_targets,), -1, dtype=torch.long, device=points.device)
    if bool(matched.any()):
        matched_positions = target_positions[matched]
        matched_indices = sorted_order[matched_positions]
        gathered_context[matched] = pooled_context[matched_indices]
        gathered_patch_ids[matched] = pooled_patch_ids[matched_indices]
    return gathered_context, gathered_patch_ids
