from ssd.modeling import registry
from .fpn import FPN

__all__ = ['build_fpn', 'FPN']

def build_fpn():
    # return registry.DECODERS[cfg.MODEL.DECODER.NAME](cfg)
    return FPN()
