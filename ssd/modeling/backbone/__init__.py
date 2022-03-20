from ssd.modeling import registry
from .vgg import VGG
from .resnet import resnet101

__all__ = ['build_backbone', 'VGG', 'resnet101']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
