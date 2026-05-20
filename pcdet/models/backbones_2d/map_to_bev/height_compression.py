import torch.nn as nn
import torch
import torch.nn.functional as F


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class GroundAwareHeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        hidden_channels = int(self.model_cfg.get('ADAPTER_HIDDEN_CHANNELS', 16))
        self.prior_key = self.model_cfg.get('GROUND_PRIOR_KEY', 'ground_bev_map')

        self.prior_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        self.residual_head = nn.Conv2d(hidden_channels, self.num_bev_features, kernel_size=1, bias=False)
        residual_scale_init = float(self.model_cfg.get('RESIDUAL_SCALE_INIT', 1e-2))
        residual_weight_std = float(self.model_cfg.get('RESIDUAL_WEIGHT_STD', 1e-3))
        gate_weight_std = float(self.model_cfg.get('GATE_WEIGHT_STD', 0.0))
        self.residual_scale = nn.Parameter(torch.tensor([residual_scale_init], dtype=torch.float32))

        if gate_weight_std > 0:
            nn.init.normal_(self.gate_head.weight, mean=0.0, std=gate_weight_std)
        else:
            nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(self.model_cfg.get('GATE_BIAS_INIT', -4.0)))
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=residual_weight_std)

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        n, c, d, h, w = spatial_features.shape
        spatial_features = spatial_features.view(n, c * d, h, w)
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        ground_bev_map = batch_dict.get(self.prior_key, None)
        if ground_bev_map is None:
            batch_dict['spatial_features'] = spatial_features
            return batch_dict

        if ground_bev_map.dim() == 3:
            ground_bev_map = ground_bev_map.unsqueeze(1)
        elif ground_bev_map.dim() != 4:
            raise ValueError(f'Unexpected ground BEV prior shape: {ground_bev_map.shape}')

        if ground_bev_map.shape[-2:] != spatial_features.shape[-2:]:
            ground_bev_map = F.interpolate(
                ground_bev_map, size=spatial_features.shape[-2:], mode='bilinear', align_corners=False
            )

        prior_features = self.prior_encoder(ground_bev_map)
        gate_map = torch.sigmoid(self.gate_head(prior_features))
        residual = self.residual_head(prior_features)

        batch_dict['ground_prior_gate_map'] = gate_map
        batch_dict['ground_prior_residual_scale'] = self.residual_scale.detach()
        batch_dict['spatial_features'] = spatial_features + self.residual_scale * gate_map * residual
        return batch_dict


class GroundDefectHeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.ground_map_key = self.model_cfg.get('GROUND_MAP_KEY', 'ground_defect_bev_map')
        self.valid_mask_key = self.model_cfg.get('VALID_MASK_KEY', 'ground_defect_valid_mask')
        self.target_key = self.model_cfg.get('TARGET_KEY', 'ground_defect_footprint_mask')
        self.concat_valid_mask = bool(self.model_cfg.get('CONCAT_VALID_MASK', True))
        self.enable_fusion = bool(self.model_cfg.get('ENABLE_FUSION', True))

        hidden_channels = list(self.model_cfg.get('DEFECT_HIDDEN_CHANNELS', [16, 32]))
        input_channels = int(self.model_cfg.get('GROUND_MAP_CHANNELS', 5))
        if self.concat_valid_mask:
            input_channels += 1

        layers = []
        prev_channels = input_channels
        for hidden_channel in hidden_channels:
            layers.extend([
                nn.Conv2d(prev_channels, hidden_channel, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ])
            prev_channels = hidden_channel
        self.defect_encoder = nn.Sequential(*layers)
        self.defect_head = nn.Conv2d(prev_channels, 1, kernel_size=1, bias=True)
        self.gate_head = nn.Sequential(
            nn.Conv2d(prev_channels + 2, prev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prev_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(prev_channels, 1, kernel_size=1, bias=True),
        )
        self.residual_head = nn.Conv2d(prev_channels, self.num_bev_features, kernel_size=1, bias=False)

        residual_scale_init = float(self.model_cfg.get('RESIDUAL_SCALE_INIT', 1e-2))
        residual_weight_std = float(self.model_cfg.get('RESIDUAL_WEIGHT_STD', 1e-3))
        self.residual_scale = nn.Parameter(torch.tensor([residual_scale_init], dtype=torch.float32))

        nn.init.zeros_(self.defect_head.weight)
        nn.init.constant_(self.defect_head.bias, float(self.model_cfg.get('DEFECT_LOGIT_BIAS_INIT', -2.0)))
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(self.model_cfg.get('GATE_BIAS_INIT', -4.0)))
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=residual_weight_std)

        self.forward_ret_dict = {}

    def _reshape_ground_tensor(self, tensor, spatial_shape, mode='bilinear'):
        if tensor is None:
            return None
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() != 4:
            raise ValueError(f'Unexpected ground defect tensor shape: {tensor.shape}')

        if tensor.shape[-2:] != spatial_shape:
            if mode == 'nearest':
                tensor = F.interpolate(tensor, size=spatial_shape, mode=mode)
            else:
                tensor = F.interpolate(tensor, size=spatial_shape, mode=mode, align_corners=False)
        return tensor

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        n, c, d, h, w = spatial_features.shape
        spatial_features = spatial_features.view(n, c * d, h, w)
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        ground_bev_map = batch_dict.get(self.ground_map_key, None)
        valid_mask = batch_dict.get(self.valid_mask_key, None)
        if ground_bev_map is None or valid_mask is None:
            self.forward_ret_dict = {}
            batch_dict['spatial_features'] = spatial_features
            return batch_dict

        ground_bev_map = self._reshape_ground_tensor(ground_bev_map, spatial_features.shape[-2:], mode='bilinear')
        valid_mask = self._reshape_ground_tensor(valid_mask, spatial_features.shape[-2:], mode='nearest')

        if self.concat_valid_mask:
            defect_input = torch.cat([ground_bev_map, valid_mask], dim=1)
        else:
            defect_input = ground_bev_map

        defect_features = self.defect_encoder(defect_input)
        defect_logits = self.defect_head(defect_features)
        defect_prob = torch.sigmoid(defect_logits)
        gate_input = torch.cat([defect_features, defect_prob, valid_mask], dim=1)
        gate_map = torch.sigmoid(self.gate_head(gate_input)) * valid_mask
        residual = self.residual_head(defect_features)

        batch_dict['ground_defect_logits'] = defect_logits
        batch_dict['ground_defect_gate_map'] = gate_map
        batch_dict['ground_defect_residual_scale'] = self.residual_scale.detach()
        if self.enable_fusion:
            batch_dict['spatial_features'] = spatial_features + self.residual_scale * gate_map * residual
        else:
            batch_dict['spatial_features'] = spatial_features

        target_map = batch_dict.get(self.target_key, None)
        if target_map is not None:
            target_map = self._reshape_ground_tensor(target_map, defect_logits.shape[-2:], mode='nearest')

        self.forward_ret_dict = {
            'defect_logits': defect_logits,
            'valid_mask': valid_mask,
            'target_map': target_map,
            'gate_map': gate_map,
            'defect_prob': defect_prob,
            'spatial_features_pre_fusion': spatial_features,
            'feature_delta': self.residual_scale * gate_map * residual,
        }
        return batch_dict

    def get_loss(self):
        if not self.model_cfg.get('AUX_LOSS_ENABLED', True):
            return None, {}

        defect_logits = self.forward_ret_dict.get('defect_logits', None)
        valid_mask = self.forward_ret_dict.get('valid_mask', None)
        target_map = self.forward_ret_dict.get('target_map', None)
        gate_map = self.forward_ret_dict.get('gate_map', None)
        defect_prob = self.forward_ret_dict.get('defect_prob', None)
        if defect_logits is None or valid_mask is None or target_map is None:
            return None, {}

        valid_weight = valid_mask.float()
        valid_denom = valid_weight.sum().clamp_min(1.0)
        raw_loss = F.binary_cross_entropy_with_logits(defect_logits, target_map, reduction='none')
        raw_loss = (raw_loss * valid_weight).sum() / valid_denom
        loss_weight = float(self.model_cfg.get('LOSS_WEIGHT', 0.1))
        weighted_loss = raw_loss * loss_weight

        target_ratio = (target_map * valid_weight).sum() / valid_denom
        prob_valid_mean = (defect_prob * valid_weight).sum() / valid_denom
        gate_valid_mean = (gate_map * valid_weight).sum() / valid_denom
        delta = self.forward_ret_dict['feature_delta']
        delta_abs_valid_mean = (
            (delta.abs() * valid_weight).sum() /
            (valid_weight.sum() * max(delta.shape[1], 1))
        )
        base_feature = self.forward_ret_dict['spatial_features_pre_fusion']
        delta_rel_l2 = delta.norm() / base_feature.norm().clamp_min(1e-6)
        target_valid_weight = target_map * valid_weight
        target_valid_denom = target_valid_weight.sum().clamp_min(1.0)
        non_target_valid_weight = (1.0 - target_map) * valid_weight
        non_target_valid_denom = non_target_valid_weight.sum().clamp_min(1.0)
        defect_prob_target_mean = (defect_prob * target_valid_weight).sum() / target_valid_denom
        defect_prob_non_target_mean = (defect_prob * non_target_valid_weight).sum() / non_target_valid_denom
        tb_dict = {
            'loss_ground_defect': raw_loss.item(),
            'loss_ground_defect_weighted': weighted_loss.item(),
            'ground_defect_valid_ratio': valid_weight.mean().item(),
            'ground_defect_target_ratio': target_ratio.item(),
            'ground_defect_prob_valid_mean': prob_valid_mean.item(),
            'ground_defect_prob_target_mean': defect_prob_target_mean.item(),
            'ground_defect_prob_non_target_mean': defect_prob_non_target_mean.item(),
            'ground_defect_gate_valid_mean': gate_valid_mean.item(),
            'ground_defect_delta_abs_valid_mean': delta_abs_valid_mean.item(),
            'ground_defect_delta_rel_l2': delta_rel_l2.item(),
            'ground_defect_residual_scale': self.residual_scale.item(),
        }
        return weighted_loss, tb_dict

    
class HeightCompression_None(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        batch_dict['spatial_features'] = batch_dict['multi_scale_2d_features']['x_conv5']
        # print('###spatial_features.shape:', batch_dict['spatial_features'].shape)
        return batch_dict
    
    
class HeightCompression_VoxelNext(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class PseudoHeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        batch_dict['spatial_features'] = batch_dict['spatial_features_2d']
        return batch_dict
