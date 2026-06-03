from collections import OrderedDict

import torch
import torch.nn as nn

try:
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
except ImportError:
    ResNet50_Weights = None
    resnet_fpn_backbone = None


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyImageBranch(nn.Module):
    """Small image feature pyramid used for smoke tests and low-memory bring-up."""

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.out_channels = int(out_channels)
        self.stage1 = nn.Sequential(
            _ConvBlock(in_channels, self.out_channels, stride=1),
            _ConvBlock(self.out_channels, self.out_channels, stride=1),
        )
        self.stage2 = nn.Sequential(
            _ConvBlock(self.out_channels, self.out_channels, stride=2),
            _ConvBlock(self.out_channels, self.out_channels, stride=1),
        )
        self.stage3 = nn.Sequential(
            _ConvBlock(self.out_channels, self.out_channels, stride=2),
            _ConvBlock(self.out_channels, self.out_channels, stride=1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        f1 = self.stage1(images)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


class ResNet50FPNImageBranch(nn.Module):
    """ResNet-50-FPN image feature pyramid backed by torchvision cached weights."""

    def __init__(
        self,
        out_channels=256,
        weights='IMAGENET1K_V2',
        trainable_layers=0,
        returned_layers=None,
        num_scales=3,
    ):
        super().__init__()
        if resnet_fpn_backbone is None or ResNet50_Weights is None:
            raise ImportError('torchvision with resnet_fpn_backbone is required for ResNet50FPNImageBranch')
        if returned_layers is None:
            returned_layers = [1, 2, 3]
        if int(out_channels) != 256:
            raise ValueError('torchvision resnet_fpn_backbone outputs 256 channels; set OUT_CHANNELS=256')

        if isinstance(weights, str):
            if weights.lower() in ('none', 'false', '0'):
                weights_enum = None
            else:
                weights_enum = getattr(ResNet50_Weights, weights)
        else:
            weights_enum = weights
        self.out_channels = 256
        self.num_scales = int(num_scales)
        self.backbone = resnet_fpn_backbone(
            backbone_name='resnet50',
            weights=weights_enum,
            trainable_layers=int(trainable_layers),
            returned_layers=list(returned_layers),
            extra_blocks=None,
        )
        self.register_buffer(
            'image_mean',
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'image_std',
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images):
        images = (images - self.image_mean.to(dtype=images.dtype)) / self.image_std.to(dtype=images.dtype)
        features = self.backbone(images)
        if isinstance(features, OrderedDict):
            features = list(features.values())
        else:
            features = list(features)
        return features[:self.num_scales]
