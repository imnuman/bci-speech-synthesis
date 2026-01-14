"""Neural Decoder Module - EEG to intent/word decoding"""

from .tcn import TemporalConvNet
from .tensorrt_infer import TensorRTDecoder

__all__ = ['TemporalConvNet', 'TensorRTDecoder']
