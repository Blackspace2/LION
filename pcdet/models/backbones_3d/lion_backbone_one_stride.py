from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from mamba_ssm import Block as MambaBlock
from torch.nn import functional as F

from ..model_utils.retnet_attn import Block as RetNetBlock
from ..model_utils.rwkv_cls import Block as RWKVBlock
from ..model_utils.vision_lstm2 import xLSTM_Block
from ..model_utils.ttt import TTTBlock
from .ground_context_utils import pool_ground_context_to_sparse_coords
from ...utils.spconv_utils import replace_feature, spconv
import torch.utils.checkpoint as cp

@torch.inference_mode()
def get_window_coors_shift_v2(coords, sparse_shape, window_shape, shift=False):
    sparse_shape_z, sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x, win_shape_y, win_shape_z = window_shape

    if shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.

    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    x = coords[:, 3] + shift_x
    y = coords[:, 2] + shift_y
    z = coords[:, 1] + shift_z

    win_coors_x = x // win_shape_x
    win_coors_y = y // win_shape_y
    win_coors_z = z // win_shape_z

    coors_in_win_x = x % win_shape_x
    coors_in_win_y = y % win_shape_y
    coors_in_win_z = z % win_shape_z

    batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + \
                       win_coors_y * max_num_win_z + win_coors_z
    batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z + \
                       win_coors_x * max_num_win_z + win_coors_z

    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds_x, batch_win_inds_y, coors_in_win


def get_window_coors_shift_v1(coords, sparse_shape, window_shape):
    _, m, n = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    x = coords[:, 3]
    y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2


class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            group_size,
            shift,
            win_version='v2'
    ) -> None:
        super().__init__()
        self.window_shape = window_shape
        self.group_size = group_size
        self.win_version = win_version
        self.shift = shift

    def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        coords = coords.long()
        actual_batch_size = max(batch_size, int(coords[:, 0].max().item()) + 1) if coords.numel() > 0 else batch_size
        num_per_batch = torch.bincount(coords[:, 0], minlength=actual_batch_size)
        flat2win_list = []
        win2flat = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        flat_start = 0
        padded_start = 0

        for i in range(actual_batch_size):
            cur_num = int(num_per_batch[i].item())
            if cur_num == 0:
                continue

            cur_flat = torch.arange(cur_num, dtype=torch.long, device=coords.device)
            padded_num = int(math.ceil(cur_num / self.group_size) * self.group_size)
            pad_num = padded_num - cur_num

            if pad_num > 0:
                repeat_times = int(math.ceil(pad_num / cur_num))
                cur_flat_padded = torch.cat([cur_flat, cur_flat.repeat(repeat_times)[:pad_num]], dim=0)
            else:
                cur_flat_padded = cur_flat

            flat2win_list.append(cur_flat_padded + flat_start)
            win2flat[flat_start: flat_start + cur_num] = torch.arange(
                cur_num, dtype=torch.long, device=coords.device
            ) + padded_start

            flat_start += cur_num
            padded_start += padded_num

        if flat_start != coords.shape[0]:
            raise RuntimeError(
                f'FlattenedWindowMapping covered {flat_start} coords, but received {coords.shape[0]}. '
                f'batch_size={batch_size}, actual_batch_size={actual_batch_size}'
            )

        flat2win = torch.cat(flat2win_list, dim=0) if flat2win_list else torch.empty(
            0, dtype=torch.long, device=coords.device
        )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}

        get_win = self.win_version

        if get_win == 'v1':
            for shifted in [False]:
                (
                    n2,
                    m2,
                    n1,
                    m1,
                    x1,
                    y1,
                    x2,
                    y2,
                ) = get_window_coors_shift_v1(coords, sparse_shape, self.window_shape)
                vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
                vx += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
                vy += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
                _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        elif get_win == 'v2':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape, self.shift)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x"] = torch.sort(vx)
            _, mappings["y"] = torch.sort(vy)

        elif get_win == 'v3':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx_xy = vx + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vx_yx = vx + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy_xy = vy + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vy_yx = vy + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x_xy"] = torch.sort(vx_xy)
            _, mappings["y_xy"] = torch.sort(vy_xy)
            _, mappings["x_yx"] = torch.sort(vx_yx)
            _, mappings["y_yx"] = torch.sort(vy_yx)

        return mappings


class PatchMerging3D(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=-1,
        down_scale=[2, 2, 2],
        norm_layer=nn.LayerNorm,
        diffusion=False,
        diff_scale=0.2,
        ground_guided_cfg=None,
        debug_name='patch_merging'
    ):
        super().__init__()
        self.dim = dim
        self.debug_name = debug_name

        self.sub_conv = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        if out_dim == -1:
            self.norm = norm_layer(dim)
        else:
            self.norm = norm_layer(out_dim)

        self.sigmoid = nn.Sigmoid()
        self.down_scale = down_scale
        self.diffusion = diffusion
        self.diff_scale = diff_scale

        self.num_points = 6 #3
        self.ground_guided_cfg = ground_guided_cfg
        self.ground_guided_enabled = ground_guided_cfg is not None and ground_guided_cfg.get('ENABLED', False)
        if self.ground_guided_enabled:
            self.prior_key = ground_guided_cfg.get('PRIOR_KEY', 'coarse_observed_ground_prior')
            self.ablation_mode = ground_guided_cfg.get('ABLATION_MODE', 'response_plus_learned_trust')
            self.pure_ground_threshold = float(ground_guided_cfg.get('PURE_GROUND_THRESHOLD', 0.95))
            self.guided_metric_interval = max(int(ground_guided_cfg.get('LOG_METRICS_EVERY', 50)), 1)
            prior_component_weights = ground_guided_cfg.get('PRIOR_COMPONENT_WEIGHTS', [1.0, 0.25, 0.25, 0.5])
            if len(prior_component_weights) != 4:
                raise ValueError(
                    f'GROUND_GUIDED_DIFFUSION PRIOR_COMPONENT_WEIGHTS expects 4 values, got {prior_component_weights}'
                )
            self.register_buffer(
                'prior_component_weights',
                torch.tensor(prior_component_weights, dtype=torch.float32).view(1, 4)
            )
            self.fixed_prior_alpha = float(ground_guided_cfg.get('FIXED_ALPHA', 0.25))
            self.alpha_decay_cfg = ground_guided_cfg.get('ALPHA_DECAY', None)
            self.response_proj = nn.Linear(dim, 1, bias=False)
            nn.init.constant_(self.response_proj.weight, 1.0 / max(float(dim), 1.0))
            feature_scale_init = max(float(ground_guided_cfg.get('DIFFUSION_FEATURE_SCALE_INIT', 1e-3)), 1e-6)
            self.diffusion_feature_scale_logit = nn.Parameter(
                torch.log(torch.expm1(torch.tensor(feature_scale_init, dtype=torch.float32)))
            )

            if self.ablation_mode == 'response_plus_learned_trust':
                alpha_init = max(float(ground_guided_cfg.get('LEARNED_ALPHA_INIT', 0.05)), 1e-4)
                self.prior_alpha_logit = nn.Parameter(torch.log(torch.expm1(torch.tensor(alpha_init, dtype=torch.float32))))
                self.prior_trust_logit = nn.Parameter(
                    torch.tensor(float(ground_guided_cfg.get('TRUST_LOGIT_INIT', -4.0)), dtype=torch.float32)
                )
            elif self.ablation_mode not in ('response_only', 'response_plus_fixed_prior'):
                raise ValueError(f'Unsupported GROUND_GUIDED_DIFFUSION ablation mode: {self.ablation_mode}')

    def _get_alpha_decay_multiplier(self, prior_context, device, dtype):
        if self.alpha_decay_cfg is None or not self.alpha_decay_cfg.get('ENABLED', False):
            return torch.ones((), device=device, dtype=dtype)

        warmup_steps = max(int(self.alpha_decay_cfg.get('WARMUP_STEPS', 0)), 1)
        global_step = max(int(prior_context.get('global_step', 0)), 0)
        start_multiplier = float(self.alpha_decay_cfg.get('START_MULTIPLIER', 0.0))
        end_multiplier = float(self.alpha_decay_cfg.get('END_MULTIPLIER', 1.0))
        progress = min(float(global_step) / float(warmup_steps), 1.0)
        multiplier = start_multiplier + (end_multiplier - start_multiplier) * progress
        return torch.tensor(multiplier, device=device, dtype=dtype)

    def _get_resized_prior_maps(self, prior_context, target_hw, device, dtype):
        if prior_context is None:
            return None
        prior_maps = prior_context.get('prior_maps', None)
        if prior_maps is None:
            return None

        cache = prior_context.setdefault('resized_prior_cache', {})
        cache_key = tuple(int(x) for x in target_hw)
        if cache_key not in cache:
            resized_prior = prior_maps
            if resized_prior.shape[-2:] != cache_key:
                resized_prior = F.interpolate(
                    resized_prior, size=cache_key, mode='bilinear', align_corners=False
                )
            cache[cache_key] = resized_prior
        return cache[cache_key].to(device=device, dtype=dtype)

    def _record_ground_guided_metrics(self, prior_context, metric_dict):
        if prior_context is None:
            return
        tb_dict = prior_context.setdefault('tb_dict', {})
        for key, value in metric_dict.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                value = float(value.detach().item())
            tb_dict[f'ground_guided/{self.debug_name}/{key}'] = float(value)

    def _should_record_guided_metrics(self, prior_context):
        if not self.ground_guided_enabled or prior_context is None:
            return False
        if not self.training:
            return True
        global_step = max(int(prior_context.get('global_step', 0)), 0)
        return (global_step % self.guided_metric_interval) == 0

    def _compute_guided_scores(self, x, prior_context):
        if not self.ground_guided_enabled:
            response_score = x.features.mean(-1)
            return response_score, response_score, None

        learned_response = self.response_proj(x.features).squeeze(-1)
        if self.ablation_mode == 'response_only':
            return learned_response, learned_response, {
                'alpha': learned_response.new_zeros(()),
                'trust': learned_response.new_zeros(()),
            }

        resized_prior_maps = self._get_resized_prior_maps(
            prior_context=prior_context,
            target_hw=(x.spatial_shape[1], x.spatial_shape[2]),
            device=x.features.device,
            dtype=x.features.dtype
        )
        if resized_prior_maps is None:
            return learned_response, learned_response, {
                'alpha': learned_response.new_zeros(()),
                'trust': learned_response.new_zeros(()),
            }

        batch_idx = x.indices[:, 0].long()
        y_idx = x.indices[:, 2].long()
        x_idx = x.indices[:, 3].long()
        prior_components = resized_prior_maps[batch_idx, :, y_idx, x_idx]
        prior_bias = (prior_components * self.prior_component_weights.to(prior_components.dtype)).sum(-1)
        prior_bias = prior_bias * prior_components[:, 1]

        if self.ablation_mode == 'response_plus_fixed_prior':
            alpha = learned_response.new_tensor(self.fixed_prior_alpha)
            trust = learned_response.new_ones(())
        else:
            alpha = F.softplus(self.prior_alpha_logit).to(device=learned_response.device, dtype=learned_response.dtype)
            trust = torch.sigmoid(self.prior_trust_logit).to(device=learned_response.device, dtype=learned_response.dtype)
            alpha = alpha * trust

        alpha = alpha * self._get_alpha_decay_multiplier(
            prior_context=prior_context,
            device=learned_response.device,
            dtype=learned_response.dtype
        )
        guided_score = learned_response + alpha * prior_bias
        return learned_response, guided_score, {
            'prior_components': prior_components,
            'prior_bias': prior_bias,
            'alpha': alpha,
            'trust': trust,
        }

    def forward(self, x, coords_shift=1, diffusion_scale=4, prior_context=None):
        assert diffusion_scale==4 or diffusion_scale==2
        x = self.sub_conv(x)

        d, h, w = x.spatial_shape
        down_scale = self.down_scale

        if self.diffusion:
            response_score, x_feat_att, guided_debug = self._compute_guided_scores(x, prior_context)
            batch_size = int(x.indices[:, 0].max().item()) + 1 if x.indices.numel() > 0 else int(x.batch_size)
            record_guided_metrics = guided_debug is not None and self._should_record_guided_metrics(prior_context)
            selected_diffusion_feats_list = [x.features]
            selected_diffusion_coords_list = [x.indices]
            topk_valid_ratio = [] if record_guided_metrics else None
            topk_boundary_ratio = [] if record_guided_metrics else None
            topk_pure_ground_ratio = [] if record_guided_metrics else None
            topk_prior_bias_mean = [] if record_guided_metrics else None
            topk_response_mean = [] if record_guided_metrics else None
            topk_guided_score_mean = [] if record_guided_metrics else None
            batch_indices = x.indices[:, 0]
            guided_prior_components = None
            guided_prior_bias = None
            if record_guided_metrics:
                guided_prior_components = guided_debug.get('prior_components', None)
                guided_prior_bias = guided_debug.get('prior_bias', None)
            diffusion_feature_scale = None
            if self.ground_guided_enabled:
                diffusion_feature_scale = F.softplus(self.diffusion_feature_scale_logit).to(
                    device=x.features.device, dtype=x.features.dtype
                )
            for i in range(batch_size):
                selected_idx = torch.nonzero(batch_indices == i, as_tuple=False).squeeze(1)
                valid_num = int(selected_idx.numel())
                if valid_num == 0:
                    continue
                K = max(1, min(valid_num, int(math.ceil(valid_num * self.diff_scale))))
                masked_scores = x_feat_att.index_select(0, selected_idx)
                if K == valid_num:
                    topk_local_indices = torch.arange(valid_num, device=selected_idx.device)
                else:
                    _, topk_local_indices = torch.topk(masked_scores, K, sorted=False)
                selected_topk_idx = selected_idx.index_select(0, topk_local_indices)

                selected_coords_copy = x.indices.index_select(0, selected_topk_idx).clone()
                selected_coords_num = selected_coords_copy.shape[0]
                selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1)
                if self.ground_guided_enabled:
                    selected_feature_scale = diffusion_feature_scale * torch.sigmoid(
                        masked_scores.index_select(0, topk_local_indices)
                    ).unsqueeze(1)
                    selected_feats = x.features.index_select(0, selected_topk_idx) * selected_feature_scale
                    selected_feats_expand = selected_feats.repeat(diffusion_scale, 1)
                else:
                    selected_feats_expand = x.features.index_select(0, selected_topk_idx).repeat(diffusion_scale, 1) * 0.0


                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (
                            selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (
                            selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 3:4] = (
                        selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 2:3] = (
                        selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                if diffusion_scale==4:
#                         print('####diffusion_scale==4')
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (
                        selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (
                        selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (
                            selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (
                            selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_diffusion_coords_list.append(selected_coords_expand)
                selected_diffusion_feats_list.append(selected_feats_expand)
                if guided_prior_components is not None:
                    top_prior_components = guided_prior_components.index_select(0, selected_topk_idx)
                    topk_valid_ratio.append(top_prior_components[:, 1].mean())
                    topk_boundary_ratio.append(top_prior_components[:, 3].mean())
                    topk_pure_ground_ratio.append(
                        ((top_prior_components[:, 0] >= self.pure_ground_threshold) & (top_prior_components[:, 1] >= 0.5)).float().mean()
                    )
                    topk_prior_bias_mean.append(guided_prior_bias.index_select(0, selected_topk_idx).mean())
                    topk_response_mean.append(response_score.index_select(0, selected_topk_idx).mean())
                    topk_guided_score_mean.append(x_feat_att.index_select(0, selected_topk_idx).mean())

            if record_guided_metrics:
                def _safe_mean(items):
                    if not items:
                        return None
                    return torch.stack(items).mean()

                input_prior_maps = None if prior_context is None else prior_context.get('prior_maps', None)
                prior_coverage = None
                if input_prior_maps is not None:
                    prior_coverage = input_prior_maps[:, 1].float().mean()
                self._record_ground_guided_metrics(prior_context, {
                    'prior_coverage': prior_coverage,
                    'topk_valid_ratio': _safe_mean(topk_valid_ratio),
                    'topk_boundary_ratio': _safe_mean(topk_boundary_ratio),
                    'topk_pure_ground_ratio': _safe_mean(topk_pure_ground_ratio),
                    'topk_prior_bias_mean': _safe_mean(topk_prior_bias_mean),
                    'topk_response_mean': _safe_mean(topk_response_mean),
                    'topk_guided_score_mean': _safe_mean(topk_guided_score_mean),
                    'diffusion_feature_scale': diffusion_feature_scale if self.ground_guided_enabled else None,
                    'alpha': guided_debug.get('alpha', None),
                    'trust': guided_debug.get('trust', None),
                })

            coords = torch.cat(selected_diffusion_coords_list)
            final_diffusion_feats = torch.cat(selected_diffusion_feats_list)

        else:
            coords = x.indices.clone()
            final_diffusion_feats = x.features.clone()

        coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
        coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
        coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        scale_xyz = new_sparse_shape[0] * new_sparse_shape[1] * new_sparse_shape[2]
        scale_yz = new_sparse_shape[0] * new_sparse_shape[1]
        scale_z = new_sparse_shape[0]


        merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        features_expand = final_diffusion_feats

        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

        x_merge = torch_scatter.scatter_add(features_expand, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        if voxel_coords.numel() > 0:
            max_zyx = voxel_coords[:, 1:].max(dim=0).values
            min_zyx = voxel_coords[:, 1:].min(dim=0).values
            shape = torch.tensor(new_sparse_shape, device=voxel_coords.device, dtype=voxel_coords.dtype)
            if (min_zyx < 0).any() or (max_zyx >= shape).any():
                raise RuntimeError(
                    f'PatchMerging3D produced out-of-range coords: '
                    f'min_zyx={min_zyx.tolist()}, max_zyx={max_zyx.tolist()}, shape={new_sparse_shape}'
                )

        x_merge = self.norm(x_merge)

        x_merge = spconv.SparseConvTensor(
            features=x_merge,
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        return x_merge, unq_inv


class PatchExpanding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, up_x, unq_inv):
        # z, y, x
        n, c = x.features.shape

        x_copy = torch.gather(x.features, 0, unq_inv.unsqueeze(1).repeat(1, c))
        up_x = up_x.replace_feature(up_x.features + x_copy)
        return up_x


LinearOperatorMap = {
    'Mamba': MambaBlock,
    'RWKV': RWKVBlock,
    'RetNet': RetNetBlock,
    'xLSTM': xLSTM_Block,
    'TTT': TTTBlock,
}


class LIONLayer(nn.Module):
    def __init__(self, dim, nums, window_shape, group_size, direction, shift, operator=None, layer_id=0, n_layer=0):
        super(LIONLayer, self).__init__()

        self.window_shape = window_shape
        self.group_size = group_size
        self.dim = dim
        self.direction = direction

        operator_cfg = operator.CFG
        operator_cfg['d_model'] = dim

        block_list = []
        for i in range(len(direction)):
            operator_cfg['layer_id'] = i + layer_id
            operator_cfg['n_layer'] = n_layer
            # operator_cfg['with_cp'] = layer_id >= 16
            operator_cfg['with_cp'] = layer_id >= 0 ## all lion layer use checkpoint to save GPU memory!! (less 24G for training all models!!!)
            print('### use part of checkpoint!!')
            block_list.append(LinearOperatorMap[operator.NAME](**operator_cfg))

        self.blocks = nn.ModuleList(block_list)
        self.window_partition = FlattenedWindowMapping(self.window_shape, self.group_size, shift)

    def forward(self, x):
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)

        for i, block in enumerate(self.blocks):
            indices = mappings[self.direction[i]]
            x_features = x.features[indices][mappings["flat2win"]]
            x_features = x_features.view(-1, self.group_size, x.features.shape[-1])

            x_features = block(x_features)

            x_features = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]
            x.features[indices] = x_features.to(dtype=x.features.dtype)

        return x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class GroundEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, norm_type='LN'):
        super().__init__()
        norm_type = str(norm_type).upper()
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm1d(hidden_dim, eps=1e-3, momentum=0.01)
        elif norm_type == 'LN':
            norm_layer = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f'Unsupported GroundEncoder norm_type: {norm_type}')

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.norm = norm_layer
        self.act = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)

        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, ground_context_raw):
        if ground_context_raw.shape[0] == 0:
            return ground_context_raw.new_zeros((0, self.linear2.out_features))

        x = self.linear1(ground_context_raw)
        x = self.norm(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class FiLMHead(nn.Module):
    def __init__(self, context_dim, feature_dim, alpha_init=0.05, weight_std=1e-3):
        super().__init__()
        self.gamma = nn.Linear(context_dim, feature_dim, bias=True)
        self.beta = nn.Linear(context_dim, feature_dim, bias=True)
        self.alpha = nn.Parameter(torch.full((1, feature_dim), float(alpha_init), dtype=torch.float32))

        nn.init.normal_(self.gamma.weight, mean=0.0, std=weight_std)
        nn.init.zeros_(self.gamma.bias)
        nn.init.normal_(self.beta.weight, mean=0.0, std=weight_std)
        nn.init.zeros_(self.beta.bias)

    def forward(self, context_features):
        gamma = self.gamma(context_features)
        beta = self.beta(context_features)
        return gamma, beta, self.alpha


class LIONBlock(nn.Module):
    def __init__(self, dim: int, depth: int, down_scales: list, window_shape, group_size, direction, shift=False,
                 operator=None, layer_id=0, n_layer=0, ground_guided_cfg=None, debug_prefix='lion_block'):
        super().__init__()

        self.down_scales = down_scales

        self.encoder = nn.ModuleList()
        self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()

        norm_fn = partial(nn.LayerNorm)

        shift = [False, shift]
        for idx in range(depth):
            self.encoder.append(LIONLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + idx * 2, n_layer))
            self.pos_emb_list.append(PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim))
            self.downsample_list.append(
                PatchMerging3D(
                    dim,
                    dim,
                    down_scale=down_scales[idx],
                    norm_layer=norm_fn,
                    ground_guided_cfg=ground_guided_cfg,
                    debug_name=f'{debug_prefix}_inner_down{idx + 1}'
                )
            )

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for idx in range(depth):
            self.decoder.append(LIONLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + 2 * (idx + depth), n_layer))
            self.decoder_norm.append(norm_fn(dim))
            
            self.upsample_list.append(PatchExpanding3D(dim))
            

    def forward(self, x, prior_context=None):
        features = []
        index = []

        for idx, enc in enumerate(self.encoder):
            pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],
                                         embed_layer=self.pos_emb_list[idx])

            x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
            x = enc(x)
            features.append(x)
            x, unq_inv = self.downsample_list[idx](x, prior_context=prior_context)
            index.append(unq_inv)

        i = 0
        for dec, norm, up_x, unq_inv, up_scale in zip(self.decoder, self.decoder_norm, features[::-1],
                                                      index[::-1], self.down_scales[::-1]):
            x = dec(x)
            x = self.upsample_list[i](x, up_x, unq_inv)
            x = replace_feature(x, norm(x.features))
            i = i + 1
        return x

    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        embed_layer = embed_layer
        if len(window_shape) == 2:
            win_x, win_y = window_shape
            win_z = 1
            use_degenerate_z = True
        elif window_shape[-1] == 1:
            win_x, win_y = window_shape[:2]
            win_z = 1
            use_degenerate_z = True
        else:
            win_x, win_y, win_z = window_shape
            use_degenerate_z = False

        z = coors[:, 0] - win_z / 2
        y = coors[:, 1] - win_y / 2
        x = coors[:, 2] - win_x / 2

        if normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            if use_degenerate_z:
                z = torch.zeros_like(x)
            else:
                z = z / win_z * 2 * 3.1415  # [-pi, pi]
        elif use_degenerate_z:
            z = torch.zeros_like(x)

        location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed
    
    

class MLPBlock(nn.Module):
    def __init__(self, input_channel, out_channel, norm_fn):
        super().__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_channel, out_channel),
            norm_fn(out_channel),
            nn.GELU())

    def forward(self, x):
        mpl_feats = self.mlp_layer(x)
        return mpl_feats

#for waymo and nuscenes, kitti, once
class LION3DBackboneOneStride(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size=None, point_cloud_range=None, point_feature_names=None, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR
        self.ground_guided_cfg = model_cfg.get('GROUND_GUIDED_DIFFUSION', None)
        self.ground_guided_enabled = self.ground_guided_cfg is not None and self.ground_guided_cfg.get('ENABLED', False)
        self.ground_context_cfg = model_cfg.get('GROUND_CONTEXT_FILM', None)
        self.ground_context_enabled = self.ground_context_cfg is not None and self.ground_context_cfg.get('ENABLED', False)
        if self.ground_guided_enabled and self.ground_context_enabled:
            raise ValueError('GROUND_GUIDED_DIFFUSION and GROUND_CONTEXT_FILM should not be enabled together')
        self.ground_guided_prior_key = 'coarse_observed_ground_prior'
        if self.ground_guided_enabled:
            self.ground_guided_prior_key = self.ground_guided_cfg.get('PRIOR_KEY', self.ground_guided_prior_key)
        self.point_feature_names = list(point_feature_names) if point_feature_names is not None else None
        self.ground_point_feature_indices = None
        if self.ground_context_enabled:
            required_names = ['is_ground', 'delta_z_to_ground', 'ground_valid']
            if self.point_feature_names is None:
                raise ValueError('GROUND_CONTEXT_FILM requires point_feature_names to be passed into the backbone')
            missing_names = [name for name in required_names if name not in self.point_feature_names]
            if missing_names:
                raise ValueError(
                    f'GROUND_CONTEXT_FILM requires point features {missing_names}, got {self.point_feature_names}'
                )
            self.ground_point_feature_indices = {
                name: self.point_feature_names.index(name) + 1 for name in required_names
            }
            if voxel_size is None or point_cloud_range is None:
                raise ValueError('GROUND_CONTEXT_FILM requires voxel_size and point_cloud_range')
            self.register_buffer('ground_context_voxel_size', torch.tensor(voxel_size, dtype=torch.float32))
            self.register_buffer('ground_context_point_cloud_range', torch.tensor(point_cloud_range, dtype=torch.float32))
        else:
            self.register_buffer('ground_context_voxel_size', torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
            self.register_buffer('ground_context_point_cloud_range', torch.tensor([0.0] * 6, dtype=torch.float32))
        
        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)

        
        self.linear_1 = LIONBlock(self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
                                    self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0,
                                    n_layer=self.n_layer, ground_guided_cfg=self.ground_guided_cfg, debug_prefix='linear_1')  ##[27, 27, 32] --》 [13, 13, 32]

        self.dow1 = PatchMerging3D(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale,
                                     ground_guided_cfg=self.ground_guided_cfg, debug_name='dow1')
        

        # [944, 944, 16] -> [472, 472, 8]
        self.linear_2 = LIONBlock(self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
                                    self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8,
                                    n_layer=self.n_layer, ground_guided_cfg=self.ground_guided_cfg, debug_prefix='linear_2')

        self.dow2 = PatchMerging3D(self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale,
                                     ground_guided_cfg=self.ground_guided_cfg, debug_name='dow2')


        #  [236, 236, 8] -> [236, 236, 4]
        self.linear_3 = LIONBlock(self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
                                    self.group_size[2], direction, shift=shift, operator=self.linear_operator, layer_id=16,
                                    n_layer=self.n_layer, ground_guided_cfg=self.ground_guided_cfg, debug_prefix='linear_3')

        self.dow3 = PatchMerging3D(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale,
                                     ground_guided_cfg=self.ground_guided_cfg, debug_name='dow3')

        #  [236, 236, 4] -> [236, 236, 2]
        self.linear_4 = LIONBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                    self.group_size[3], direction, shift=shift, operator=self.linear_operator, layer_id=24,
                                    n_layer=self.n_layer, ground_guided_cfg=self.ground_guided_cfg, debug_prefix='linear_4')

        self.dow4 = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale,
                                     ground_guided_cfg=self.ground_guided_cfg, debug_name='dow4')

        self.linear_out = LIONLayer(self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)

        self.stage_strides_xyz = [
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 4],
            [1, 1, 8],
        ]
        self.ground_context_metric_interval = 50
        if self.ground_context_enabled:
            context_dim = int(self.ground_context_cfg.get('CONTEXT_DIM', dim))
            hidden_dim = int(self.ground_context_cfg.get('ENCODER_HIDDEN_DIM', max(context_dim, 32)))
            norm_type = self.ground_context_cfg.get('ENCODER_NORM', 'LN')
            alpha_init = float(self.ground_context_cfg.get('ALPHA_INIT', 0.05))
            film_weight_std = float(self.ground_context_cfg.get('FILM_WEIGHT_STD', 1e-3))
            mode = self.ground_context_cfg.get('MODE', None)
            if mode is None:
                mode = 'full' if bool(self.ground_context_cfg.get('ENABLE_FILM', True)) else 'off'
            self.ground_context_mode = str(mode).lower()
            if self.ground_context_mode not in ('full', 'beta_only', 'off'):
                raise ValueError(
                    f'GROUND_CONTEXT_FILM.MODE expects one of full/beta_only/off, got {self.ground_context_mode}'
                )
            self.ground_context_metric_interval = max(
                int(self.ground_context_cfg.get('LOG_METRICS_EVERY', 50)), 1
            )
            self.ground_encoder = GroundEncoder(
                input_dim=4,
                hidden_dim=hidden_dim,
                output_dim=context_dim,
                norm_type=norm_type
            )
            self.film_heads = nn.ModuleList([
                FiLMHead(context_dim=context_dim, feature_dim=self.layer_dim[0], alpha_init=alpha_init, weight_std=film_weight_std),
                FiLMHead(context_dim=context_dim, feature_dim=self.layer_dim[1], alpha_init=alpha_init, weight_std=film_weight_std),
                FiLMHead(context_dim=context_dim, feature_dim=self.layer_dim[2], alpha_init=alpha_init, weight_std=film_weight_std),
                FiLMHead(context_dim=context_dim, feature_dim=self.layer_dim[3], alpha_init=alpha_init, weight_std=film_weight_std),
            ])
        else:
            self.ground_context_mode = 'off'

        self.num_point_features = dim

        self.backbone_channels = {
            'x_conv1': 128,
            'x_conv2': 128,
            'x_conv3': 128,
            'x_conv4': 128
        }

    def build_ground_context_state(self, batch_dict):
        if not self.ground_context_enabled:
            return None
        points = batch_dict.get('points', None)
        if points is None:
            raise KeyError('GROUND_CONTEXT_FILM requires batch_dict["points"]')
        return {
            'points': points,
            'tb_dict': {},
            'global_step': int(batch_dict.get('global_step', 0)),
        }

    def should_record_ground_context_metrics(self, ground_context_state):
        if not self.ground_context_enabled or ground_context_state is None:
            return False
        if not self.training:
            return True
        return (ground_context_state['global_step'] % self.ground_context_metric_interval) == 0

    def pool_ground_context_raw(self, points, target_coords, target_spatial_shape, stride_xyz):
        return pool_ground_context_to_sparse_coords(
            points=points,
            target_coords=target_coords,
            target_spatial_shape=target_spatial_shape,
            voxel_size=self.ground_context_voxel_size,
            point_cloud_range=self.ground_context_point_cloud_range,
            stride_xyz=stride_xyz,
            feature_indices=self.ground_point_feature_indices
        )

    def apply_ground_context_film(self, x, stage_idx, ground_context_raw, ground_context_state=None):
        if not self.ground_context_enabled:
            return x

        context_features = self.ground_encoder(ground_context_raw)
        gamma, beta, alpha = self.film_heads[stage_idx](context_features)
        if self.ground_context_mode == 'full':
            modulated = x.features * (1.0 + alpha * torch.tanh(gamma)) + alpha * beta
        elif self.ground_context_mode == 'beta_only':
            modulated = x.features + alpha * beta
        else:
            modulated = x.features

        if self.should_record_ground_context_metrics(ground_context_state):
            valid_ratio = ground_context_raw[:, 3] if ground_context_raw.shape[0] > 0 else modulated.new_zeros((0,))
            ground_ratio = ground_context_raw[:, 2] if ground_context_raw.shape[0] > 0 else modulated.new_zeros((0,))
            delta_feature = modulated - x.features
            stage_name = f'stage{stage_idx + 1}'
            tb_dict = ground_context_state['tb_dict']
            tb_dict[f'ground_context/{stage_name}/valid_ratio_mean'] = float(
                valid_ratio.mean().detach().item()
            ) if valid_ratio.numel() > 0 else 0.0
            tb_dict[f'ground_context/{stage_name}/ground_ratio_mean'] = float(
                ground_ratio.mean().detach().item()
            ) if ground_ratio.numel() > 0 else 0.0
            tb_dict[f'ground_context/{stage_name}/context_norm_mean'] = float(
                context_features.norm(dim=1).mean().detach().item()
            ) if context_features.numel() > 0 else 0.0
            tb_dict[f'ground_context/{stage_name}/alpha_abs_mean'] = float(alpha.abs().mean().detach().item())
            tb_dict[f'ground_context/{stage_name}/gamma_abs_mean'] = float(gamma.abs().mean().detach().item()) if gamma.numel() > 0 else 0.0
            tb_dict[f'ground_context/{stage_name}/beta_abs_mean'] = float(beta.abs().mean().detach().item()) if beta.numel() > 0 else 0.0
            tb_dict[f'ground_context/{stage_name}/delta_rel_l2'] = float(
                (delta_feature.norm() / x.features.norm().clamp_min(1e-6)).detach().item()
            ) if x.features.numel() > 0 else 0.0
            mode_value = {'off': 0.0, 'beta_only': 1.0, 'full': 2.0}[self.ground_context_mode]
            tb_dict[f'ground_context/{stage_name}/mode_id'] = mode_value

        return replace_feature(x, modulated)

    def build_ground_guided_prior_context(self, batch_dict):
        if not self.ground_guided_enabled:
            return None

        prior_maps = batch_dict.get(self.ground_guided_prior_key, None)
        if prior_maps is None:
            return None
        if prior_maps.dim() != 4 or prior_maps.shape[1] != 4:
            raise ValueError(
                f'{self.ground_guided_prior_key} expects shape [B, 4, H, W], got {tuple(prior_maps.shape)}'
            )
        return {
            'global_step': int(batch_dict.get('global_step', 0)),
            'prior_maps': prior_maps,
            'tb_dict': {},
            'resized_prior_cache': {},
        }

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        prior_context = self.build_ground_guided_prior_context(batch_dict)
        ground_context_state = self.build_ground_context_state(batch_dict)

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        if self.ground_context_enabled:
            stage0_raw = batch_dict.get('voxel_ground_context_raw', None)
            if stage0_raw is None or stage0_raw.shape[0] != voxel_coords.shape[0]:
                stage0_raw = self.pool_ground_context_raw(
                    points=ground_context_state['points'],
                    target_coords=x.indices,
                    target_spatial_shape=x.spatial_shape,
                    stride_xyz=self.stage_strides_xyz[0]
                )
            x = self.apply_ground_context_film(x, 0, stage0_raw, ground_context_state=ground_context_state)
        x = self.linear_1(x, prior_context=prior_context)
        x1, _ = self.dow1(x, prior_context=prior_context)  ## 14.0k --> 16.9k  [32, 1000, 1000]-->[16, 1000, 1000]
        if self.ground_context_enabled:
            stage1_raw = self.pool_ground_context_raw(
                points=ground_context_state['points'],
                target_coords=x1.indices,
                target_spatial_shape=x1.spatial_shape,
                stride_xyz=self.stage_strides_xyz[1]
            )
            x1 = self.apply_ground_context_film(x1, 1, stage1_raw, ground_context_state=ground_context_state)
        x = self.linear_2(x1, prior_context=prior_context)
        x2, _ = self.dow2(x, prior_context=prior_context)  ## 16.9k --> 18.8k  [16, 1000, 1000]-->[8, 1000, 1000]
        if self.ground_context_enabled:
            stage2_raw = self.pool_ground_context_raw(
                points=ground_context_state['points'],
                target_coords=x2.indices,
                target_spatial_shape=x2.spatial_shape,
                stride_xyz=self.stage_strides_xyz[2]
            )
            x2 = self.apply_ground_context_film(x2, 2, stage2_raw, ground_context_state=ground_context_state)
        x = self.linear_3(x2, prior_context=prior_context)
        x3, _ = self.dow3(x, prior_context=prior_context)   ## 18.8k --> 19.1k  [8, 1000, 1000]-->[4, 1000, 1000]
        if self.ground_context_enabled:
            stage3_raw = self.pool_ground_context_raw(
                points=ground_context_state['points'],
                target_coords=x3.indices,
                target_spatial_shape=x3.spatial_shape,
                stride_xyz=self.stage_strides_xyz[3]
            )
            x3 = self.apply_ground_context_film(x3, 3, stage3_raw, ground_context_state=ground_context_state)
        x = self.linear_4(x3, prior_context=prior_context)
        x4, _ = self.dow4(x, prior_context=prior_context)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]
        x = self.linear_out(x4)

        batch_dict.update({
            'encoded_spconv_tensor': x,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x1,
                'x_conv2': x2,
                'x_conv3': x3,
                'x_conv4': x4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': torch.tensor([1,1,2], device=x1.features.device).float(),
                'x_conv2': torch.tensor([1,1,4], device=x1.features.device).float(),
                'x_conv3': torch.tensor([1,1,8], device=x1.features.device).float(),
                'x_conv4': torch.tensor([1,1,16], device=x1.features.device).float(),
            }
        })
        if prior_context is not None and prior_context['tb_dict']:
            batch_dict['ground_guided_tb_dict'] = prior_context['tb_dict']
        if ground_context_state is not None and ground_context_state['tb_dict']:
            batch_dict['ground_context_tb_dict'] = ground_context_state['tb_dict']

        return batch_dict



#for argoverse
class LION3DBackboneOneStride_Sparse(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR
        
        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2 + 2*3

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)

        
        self.linear_1 = LIONBlock(self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
                                    self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0, n_layer=self.n_layer)  ##[27, 27, 32] --》 [13, 13, 32]

        self.dow1 = PatchMerging3D(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)
        

        # [944, 944, 16] -> [472, 472, 8]
        self.linear_2 = LIONBlock(self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
                                    self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8, n_layer=self.n_layer)

        self.dow2 = PatchMerging3D(self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)


        #  [236, 236, 8] -> [236, 236, 4]
        self.linear_3 = LIONBlock(self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
                                    self.group_size[2], direction, shift=shift, operator=self.linear_operator, layer_id=16, n_layer=self.n_layer)

        self.dow3 = PatchMerging3D(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        #  [236, 236, 4] -> [236, 236, 2]
        self.linear_4 = LIONBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                    self.group_size[3], direction, shift=shift, operator=self.linear_operator, layer_id=24, n_layer=self.n_layer)

        self.dow4 = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_out = LIONLayer(self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)
        
        self.dow_out = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                        norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_bev1 = LIONLayer(self.layer_dim[3], 1, [25, 25, 1], 512, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=34, n_layer=self.n_layer)
        self.linear_bev2 = LIONLayer(self.layer_dim[3], 1, [37, 37, 1], 1024, direction=['x', 'y'], shift=shift,
                                       operator=self.linear_operator, layer_id=36, n_layer=self.n_layer)
        self.linear_bev3 = LIONLayer(self.layer_dim[3], 1, [51, 51, 1], 2048, direction=['x', 'y'], shift=shift,
                                       operator=self.linear_operator, layer_id=38, n_layer=self.n_layer)
        

        self.num_point_features = dim

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.linear_1(x)
        x, _ = self.dow1(x)
        x = self.linear_2(x)
        x, _ = self.dow2(x)
        x = self.linear_3(x)
        x, _ = self.dow3(x)
        x = self.linear_4(x)
        x, _ = self.dow4(x)
        x = self.linear_out(x)
        
        
        x, _ = self.dow_out(x)

        x = self.linear_bev1(x)
        x = self.linear_bev2(x)
        x = self.linear_bev3(x)

        x_new = spconv.SparseConvTensor(
            features=x.features,
            indices=x.indices[:, [0, 2, 3]].type(torch.int32), #x.indices,
            spatial_shape=x.spatial_shape[1:],
            batch_size=x.batch_size
        )

        batch_dict.update({
            'encoded_spconv_tensor': x_new,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({'spatial_features_2d': x_new})

        return batch_dict
