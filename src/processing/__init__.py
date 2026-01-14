"""Signal Processing Module - Real-time EEG filtering and feature extraction"""

from .filters import SignalProcessor, NotchFilter, BandpassFilter
from .features import FeatureExtractor

__all__ = ['SignalProcessor', 'NotchFilter', 'BandpassFilter', 'FeatureExtractor']
