from ssd.modeling import registry
from .vgg import VGG
from .resnet import resnet101
from .mvgg import MVGG

__all__ = ['build_backbone', 'VGG', 'resnet101', 'MVGG']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
