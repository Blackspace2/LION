import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelAlign(nn.Module):
    """Project voxel centers to image feature maps and sample per-voxel image vectors."""

    def __init__(self, in_channels, out_channels=64, voxel_size=None, point_cloud_range=None, image_shape=None):
        super().__init__()
        if voxel_size is None:
            voxel_size = [1.0, 1.0, 1.0]
        if point_cloud_range is None:
            point_cloud_range = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        if image_shape is None:
            image_shape = [1, 1]
        self.in_channels = [int(x) for x in in_channels]
        self.out_channels = int(out_channels)
        self.register_buffer('voxel_size', torch.tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer('point_cloud_range', torch.tensor(point_cloud_range, dtype=torch.float32), persistent=False)
        self.register_buffer('default_image_shape', torch.tensor(image_shape, dtype=torch.float32), persistent=False)

        total_channels = sum(self.in_channels)
        self.proj = nn.Sequential(
            nn.LayerNorm(total_channels),
            nn.Linear(total_channels, self.out_channels),
            nn.LayerNorm(self.out_channels),
        )

    def _effective_voxel_size(self, coords, voxel_size=None, coord_stride=None):
        device = coords.device
        dtype = torch.float32
        if voxel_size is None:
            size = self.voxel_size.to(device=device, dtype=dtype)
        else:
            size = torch.as_tensor(voxel_size, dtype=dtype, device=device)
        if coord_stride is None:
            return size
        stride = torch.as_tensor(coord_stride, dtype=dtype, device=device)
        if stride.numel() == 1:
            stride = stride.repeat(3)
        return size * stride.view(3)

    def _voxel_centers(self, coords, voxel_size=None, coord_stride=None):
        if coords.dim() != 2 or coords.shape[1] not in (3, 4):
            raise ValueError('coords must have shape (N, 4) [b,z,y,x] or (N, 3) [z,y,x]')
        if coords.shape[1] == 4:
            zyx = coords[:, 1:].to(dtype=torch.float32)
        else:
            zyx = coords.to(dtype=torch.float32)
        effective_voxel_size = self._effective_voxel_size(coords, voxel_size=voxel_size, coord_stride=coord_stride)
        point_cloud_range = self.point_cloud_range.to(device=coords.device, dtype=torch.float32)
        x = point_cloud_range[0] + (zyx[:, 2] + 0.5) * effective_voxel_size[0]
        y = point_cloud_range[1] + (zyx[:, 1] + 0.5) * effective_voxel_size[1]
        z = point_cloud_range[2] + (zyx[:, 0] + 0.5) * effective_voxel_size[2]
        return torch.stack((x, y, z), dim=-1)

    @staticmethod
    def _batch_indices(coords):
        if coords.shape[1] == 4:
            return coords[:, 0].long()
        return coords.new_zeros(coords.shape[0], dtype=torch.long)

    def _image_shapes_for_tokens(self, image_shape, batch_idx):
        if image_shape is None:
            shape = self.default_image_shape.to(device=batch_idx.device)
        else:
            shape = torch.as_tensor(image_shape, device=batch_idx.device, dtype=torch.float32)
        if shape.dim() == 1:
            height = shape[0].expand(batch_idx.shape[0])
            width = shape[1].expand(batch_idx.shape[0])
            return height, width
        if shape.dim() == 2:
            if batch_idx.numel() > 0 and int(batch_idx.max().item()) >= shape.shape[0]:
                raise ValueError('image_shape batch dimension is smaller than coords batch indices')
            selected = shape[batch_idx.long()]
            return selected[:, 0], selected[:, 1]
        raise ValueError('image_shape must have shape (2,) or (B, 2)')

    def _project(self, centers, batch_idx, lidar_to_cam, cam_to_img, image_shape):
        device = centers.device
        dtype = centers.dtype
        ones = torch.ones((centers.shape[0], 1), dtype=dtype, device=device)
        centers_h = torch.cat((centers, ones), dim=-1)
        cam_points = torch.empty_like(centers_h)
        uvw = torch.empty((centers.shape[0], 3), dtype=dtype, device=device)

        for batch_id in batch_idx.unique(sorted=True):
            mask = batch_idx == batch_id
            b = int(batch_id.item())
            cam_points[mask] = centers_h[mask].matmul(lidar_to_cam[b].to(dtype=dtype).transpose(0, 1))
            uvw[mask] = cam_points[mask].matmul(cam_to_img[b].to(dtype=dtype).transpose(0, 1))

        denom = uvw[:, 2]
        valid_denom = denom.abs() > 1e-6
        safe_denom = torch.where(valid_denom, denom, torch.ones_like(denom))
        u = uvw[:, 0] / safe_denom
        v = uvw[:, 1] / safe_denom
        height, width = self._image_shapes_for_tokens(image_shape, batch_idx)
        fov = (cam_points[:, 2] > 0) & valid_denom & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        return u, v, fov

    def _apply_inverse_aug(self, centers, batch_idx, lidar_aug_matrix):
        if lidar_aug_matrix is None:
            return centers
        device = centers.device
        dtype = centers.dtype
        ones = torch.ones((centers.shape[0], 1), dtype=dtype, device=device)
        centers_h = torch.cat((centers, ones), dim=-1)
        restored = torch.empty_like(centers_h)
        inv_aug = torch.inverse(lidar_aug_matrix.to(device=device, dtype=dtype))
        for batch_id in batch_idx.unique(sorted=True):
            mask = batch_idx == batch_id
            b = int(batch_id.item())
            restored[mask] = centers_h[mask].matmul(inv_aug[b].transpose(0, 1))
        return restored[:, :3]

    def forward(
        self,
        coords,
        image_features,
        lidar_to_cam,
        cam_to_img,
        image_shape=None,
        lidar_aug_matrix=None,
        voxel_size=None,
        coord_stride=None,
    ):
        if len(image_features) != len(self.in_channels):
            raise ValueError('image_features length must match in_channels')
        if coords.numel() == 0:
            empty = coords.new_zeros((0, self.out_channels), dtype=torch.float32)
            return empty, coords.new_zeros((0,), dtype=torch.bool)

        batch_idx = self._batch_indices(coords)
        centers = self._voxel_centers(coords, voxel_size=voxel_size, coord_stride=coord_stride).to(device=image_features[0].device)
        batch_idx = batch_idx.to(device=centers.device)
        centers = self._apply_inverse_aug(centers, batch_idx, lidar_aug_matrix)
        u, v, fov = self._project(centers, batch_idx, lidar_to_cam.to(centers.device), cam_to_img.to(centers.device), image_shape)
        height, width = self._image_shapes_for_tokens(image_shape, batch_idx)
        u_norm = (u / (width - 1.0).clamp_min(1.0)) * 2.0 - 1.0
        v_norm = (v / (height - 1.0).clamp_min(1.0)) * 2.0 - 1.0

        sampled_scales = []
        batch_size = image_features[0].shape[0]
        for feat, expected_channels in zip(image_features, self.in_channels):
            if feat.shape[1] != expected_channels:
                raise ValueError('image feature channel mismatch')
            sampled = feat.new_zeros((coords.shape[0], feat.shape[1]))
            for b in range(batch_size):
                mask = batch_idx == b
                if not mask.any():
                    continue
                grid = torch.stack((u_norm[mask], v_norm[mask]), dim=-1).view(1, -1, 1, 2)
                values = F.grid_sample(
                    feat[b:b + 1],
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                )
                sampled[mask] = values.squeeze(0).squeeze(-1).transpose(0, 1)
            sampled_scales.append(sampled)

        v_img = self.proj(torch.cat(sampled_scales, dim=-1))
        v_img = v_img * fov.to(dtype=v_img.dtype).unsqueeze(-1)
        return v_img, fov
