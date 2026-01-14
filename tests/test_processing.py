"""Tests for signal processing pipeline"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from processing.filters import SignalProcessor, NotchFilter, BandpassFilter
from processing.features import FeatureExtractor


class TestNotchFilter:
    """Tests for NotchFilter class."""

    def test_initialization(self):
        """Test filter initialization."""
        notch = NotchFilter(sample_rate=1000, freq=50)
        assert notch.sample_rate == 1000
        assert notch.freq == 50
        assert len(notch.filters) == 3  # 50, 100, 150 Hz harmonics

    def test_removes_50hz(self):
        """Test that 50Hz is attenuated."""
        sample_rate = 1000
        duration = 2.0
        channels = 8

        notch = NotchFilter(sample_rate=sample_rate, freq=50)
        notch.initialize(channels)

        # Generate 50Hz sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal_50hz = np.sin(2 * np.pi * 50 * t)
        data = np.tile(signal_50hz, (channels, 1)).T

        # Filter
        filtered = notch.process(data)

        # 50Hz power should be reduced
        assert np.std(filtered) < np.std(data) * 0.3


class TestBandpassFilter:
    """Tests for BandpassFilter class."""

    def test_initialization(self):
        """Test filter initialization."""
        bp = BandpassFilter(sample_rate=1000, low_freq=1, high_freq=50)
        assert bp.sample_rate == 1000

    def test_passes_in_band(self):
        """Test that in-band frequencies pass through."""
        sample_rate = 1000
        duration = 2.0
        channels = 4

        bp = BandpassFilter(sample_rate=sample_rate, low_freq=5, high_freq=45)
        bp.initialize(channels)

        # Generate 10Hz signal (in-band)
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal_10hz = np.sin(2 * np.pi * 10 * t)
        data = np.tile(signal_10hz, (channels, 1)).T

        filtered = bp.process(data)

        # Signal should be preserved (allowing for edge effects)
        correlation = np.corrcoef(data[500:-500, 0], filtered[500:-500, 0])[0, 1]
        assert correlation > 0.9

    def test_attenuates_out_of_band(self):
        """Test that out-of-band frequencies are attenuated."""
        sample_rate = 1000
        duration = 2.0
        channels = 4

        bp = BandpassFilter(sample_rate=sample_rate, low_freq=5, high_freq=45)
        bp.initialize(channels)

        # Generate 100Hz signal (out-of-band)
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal_100hz = np.sin(2 * np.pi * 100 * t)
        data = np.tile(signal_100hz, (channels, 1)).T

        filtered = bp.process(data)

        # Signal should be attenuated
        assert np.std(filtered) < np.std(data) * 0.3


class TestSignalProcessor:
    """Tests for SignalProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = SignalProcessor(
            sample_rate=1000,
            notch_freq=50,
            bandpass=(1, 50)
        )
        assert processor.sample_rate == 1000

    def test_process_shape(self):
        """Test output shape matches input."""
        processor = SignalProcessor(sample_rate=1000)

        data = np.random.randn(1000, 32)
        processed = processor.process(data)

        assert processed.shape == data.shape

    def test_artifact_rejection(self):
        """Test artifact rejection marks large values."""
        processor = SignalProcessor(
            sample_rate=1000,
            artifact_threshold=50
        )

        # Data with artifact
        data = np.random.randn(1000, 32) * 10
        data[500, 0] = 200  # Artifact

        processed = processor.process(data)

        # Artifact should be marked as NaN
        assert np.isnan(processed[500, 0])


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = FeatureExtractor(sample_rate=1000)
        assert extractor.sample_rate == 1000

    def test_band_power_extraction(self):
        """Test band power feature extraction."""
        extractor = FeatureExtractor(sample_rate=1000)

        data = np.random.randn(1000, 32)
        band_power = extractor.extract_band_power(data)

        assert 'alpha' in band_power
        assert 'beta' in band_power
        assert 'high_gamma' in band_power
        assert band_power['alpha'].shape == (32,)

    def test_extract_all(self):
        """Test full feature extraction."""
        extractor = FeatureExtractor(sample_rate=1000)

        data = np.random.randn(1000, 32)
        features = extractor.extract_all(data)

        # Should be a 1D vector
        assert features.ndim == 1
        assert len(features) > 0

    def test_feature_names(self):
        """Test feature name generation."""
        extractor = FeatureExtractor(sample_rate=1000)
        names = extractor.get_feature_names(channels=32)

        assert len(names) > 0
        assert 'alpha_ch0' in names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
