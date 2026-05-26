import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import ops

from ...utils.spconv_utils import replace_feature


class SBSD(nn.Module):
    def __init__(self, dim, cfg=None, point_cloud_range=None):
        super().__init__()
        cfg = cfg or {}

        self.enabled = bool(cfg.get('ENABLED', False))
        self.inject_scope = str(cfg.get('INJECT_SCOPE', 'inner_down_only'))
        self.descriptor_dim = int(cfg.get('DESCRIPTOR_DIM', 7))
        self.sigma = float(cfg.get('SIGMA', 1.0))
        self.init_scheme = str(cfg.get('INIT_SCHEME', 'zero')).lower()
        self.proj_init_std = float(cfg.get('PROJ_INIT_STD', 1e-3))
        self.proj_scale_init = float(cfg.get('PROJ_SCALE_INIT', 1e-3))
        self.fixed_proj_scale = float(cfg.get('FIXED_PROJ_SCALE', 1.0))
        self.use_bandwidth = bool(cfg.get('USE_BANDWIDTH', True))
        self.use_spectral = bool(cfg.get('USE_SPECTRAL', True))
        self.use_density = bool(cfg.get('USE_DENSITY', True))
        self.use_spectral_norm = bool(cfg.get('USE_SPECTRAL_NORM', False))
        self.spectral_norm_eps = float(cfg.get('SPECTRAL_NORM_EPS', 1e-5))
        if point_cloud_range is None:
            point_cloud_range = [0.0] * 6
        self.register_buffer('point_cloud_range', torch.tensor(point_cloud_range, dtype=torch.float32))

        self.spectral_block_dim = self.descriptor_dim - 1
        self.spectral_norm = nn.LayerNorm(self.spectral_block_dim, eps=self.spectral_norm_eps)
        self.proj = nn.Linear(self.descriptor_dim, dim, bias=False)
        if self.init_scheme == 'gaussian':
            nn.init.normal_(self.proj.weight, mean=0.0, std=self.proj_init_std)
            self.register_buffer('proj_scale', torch.tensor(self.fixed_proj_scale, dtype=torch.float32))
        elif self.init_scheme == 'zero':
            nn.init.zeros_(self.proj.weight)
            self.proj_scale = nn.Parameter(torch.tensor(self.proj_scale_init, dtype=torch.float32))
        else:
            raise ValueError(f'Unsupported SBSD INIT_SCHEME: {self.init_scheme}')

    def _get_voxel_size(self, x):
        spatial_shape = x.features.new_tensor(list(x.spatial_shape), dtype=x.features.dtype)
        extent = self.point_cloud_range[3:] - self.point_cloud_range[:3]
        extent = extent.to(device=x.features.device, dtype=x.features.dtype)
        # point_cloud_range is xyz while sparse coords are zyx.
        extent_zyx = extent[[2, 1, 0]]
        return extent_zyx / spatial_shape.clamp(min=1)

    def _get_out_to_input(self, x, out_indices):
        if out_indices.shape[0] == x.indices.shape[0] and torch.equal(out_indices, x.indices):
            return torch.arange(x.indices.shape[0], device=x.indices.device, dtype=torch.long)

        spatial_shape = [int(v) for v in x.spatial_shape]
        batch_stride = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        z_stride = spatial_shape[1] * spatial_shape[2]
        y_stride = spatial_shape[2]

        def encode(indices):
            return (
                indices[:, 0].long() * batch_stride
                + indices[:, 1].long() * z_stride
                + indices[:, 2].long() * y_stride
                + indices[:, 3].long()
            )

        input_code = encode(x.indices)
        out_code = encode(out_indices)
        sorted_input_code, sorted_input_order = torch.sort(input_code)
        positions = torch.searchsorted(sorted_input_code, out_code)
        matched_order = sorted_input_order[positions]
        if not torch.equal(input_code[matched_order], out_code):
            raise RuntimeError('SBSD expected submanifold output coords to match input coords')
        return matched_order

    def _get_subm_pair(self, x):
        pair = ops.get_indice_pairs_implicit_gemm(
            x.indices,
            batch_size=int(x.batch_size),
            spatial_shape=[int(v) for v in x.spatial_shape],
            algo=spconv.ConvAlgo.MaskImplicitGemm,
            ksize=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            dilation=[1, 1, 1],
            out_padding=[0, 0, 0],
            subm=True,
            transpose=False,
            is_train=self.training,
        )
        return {
            'out_indices': pair[0],
            'indice_num_per_loc': pair[1],
            'pair_fwd': pair[2],
            'pair_bwd': pair[3],
            'pair_mask_fwd_splits': pair[4],
            'pair_mask_bwd_splits': pair[5],
            'mask_argsort_fwd_splits': pair[6],
            'mask_argsort_bwd_splits': pair[7],
            'masks': pair[8],
        }

    def _compute_descriptor(self, x):
        if x.features.shape[0] == 0:
            return x.features.new_zeros((0, self.descriptor_dim))

        pair = self._get_subm_pair(x)
        out_indices = pair['out_indices']
        pair_fwd = pair['pair_fwd'].long()
        out_to_input = self._get_out_to_input(x, out_indices)

        voxel_size = self._get_voxel_size(x)
        input_pos = x.indices[:, 1:].to(dtype=x.features.dtype) * voxel_size.unsqueeze(0)
        out_pos = out_indices[:, 1:].to(dtype=x.features.dtype) * voxel_size.unsqueeze(0)

        clamped_pair = pair_fwd.clamp(min=0)
        valid = pair_fwd >= 0
        neighbor_pos = input_pos[clamped_pair]
        center_pos = out_pos.unsqueeze(0)
        delta = neighbor_pos - center_pos
        dist2 = delta.square().sum(-1)

        sigma_sq = max(self.sigma * self.sigma, 1e-6)
        weights = torch.exp(-dist2 / (2.0 * sigma_sq)) * valid.to(dtype=x.features.dtype)
        weights = weights * (dist2 > 0).to(dtype=x.features.dtype)

        rho = weights.sum(0)
        rho_safe = rho.clamp(min=1e-6)
        dp_out = (weights.unsqueeze(-1) * delta).sum(0) / rho_safe.unsqueeze(-1)

        laplace_pos_out = (weights.unsqueeze(-1) * (center_pos - neighbor_pos)).sum(0)
        laplace_pos_input = input_pos.new_zeros(input_pos.shape)
        laplace_pos_input[out_to_input] = laplace_pos_out

        neighbor_laplace = laplace_pos_input[clamped_pair]
        center_laplace = laplace_pos_out.unsqueeze(0)
        laplace2_pos_out = (weights.unsqueeze(-1) * (center_laplace - neighbor_laplace)).sum(0)

        e1 = laplace_pos_out.norm(dim=-1, keepdim=True)
        e2 = laplace2_pos_out.norm(dim=-1, keepdim=True)

        if not self.use_spectral:
            e1.zero_()
            e2.zero_()
        if not self.use_density:
            rho.zero_()
            dp_out.zero_()

        bbar = rho.unsqueeze(-1).new_zeros((rho.shape[0], 1))
        if self.use_bandwidth:
            input_rank = torch.arange(
                x.indices.shape[0],
                device=x.indices.device,
                dtype=x.features.dtype,
            )
            rank_scale = max(x.indices.shape[0] - 1, 1)
            input_rank = input_rank / rank_scale
            center_rank = input_rank[out_to_input].unsqueeze(0)
            neighbor_rank = input_rank[clamped_pair]
            bbar = (
                (weights * (center_rank - neighbor_rank).abs()).sum(0) / rho_safe
            ).unsqueeze(-1).detach()
        else:
            bbar.zero_()

        spectral_block = torch.cat([e1, e2, rho.unsqueeze(-1), dp_out], dim=-1)
        if self.use_spectral_norm and (self.use_spectral or self.use_density):
            spectral_block = self.spectral_norm(spectral_block)

        descriptor_out = torch.cat([bbar, spectral_block], dim=-1)
        descriptor_input = x.features.new_zeros((x.features.shape[0], descriptor_out.shape[-1]))
        descriptor_input[out_to_input] = descriptor_out
        return descriptor_input

    def forward(self, x):
        if not self.enabled:
            return x

        descriptor = self._compute_descriptor(x)
        delta_feature = self.proj(descriptor) * self.proj_scale.to(device=x.features.device, dtype=x.features.dtype)
        return replace_feature(x, x.features + delta_feature)
