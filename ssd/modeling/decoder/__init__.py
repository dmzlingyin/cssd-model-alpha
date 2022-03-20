from ssd.modeling import registry
from .decoder import SSDDecoder

__all__ = ['build_decoder', 'SSDDecoder']

def build_decoder(cfg):
    return registry.DECODERS[cfg.MODEL.DECODER.NAME](cfg)
