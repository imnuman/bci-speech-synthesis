"""
Feature Extraction for EEG-based Speech Decoding
Extracts band power and spectral features for neural decoding
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FrequencyBands:
    """Standard EEG frequency bands."""
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    low_gamma: Tuple[float, float] = (30.0, 50.0)
    high_gamma: Tuple[float, float] = (50.0, 100.0)  # Important for speech


class FeatureExtractor:
    """
    Extract features from EEG data for neural decoding.

    Features extracted:
    1. Band power (log power in frequency bands)
    2. Spectral entropy
    3. Hjorth parameters (activity, mobility, complexity)
    4. Statistical moments

    High Gamma (50-100 Hz) is particularly important for speech decoding
    as it correlates with neural activity in speech production areas.
    """

    def __init__(
        self,
        sample_rate: float = 1000.0,
        bands: Optional[FrequencyBands] = None
    ):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Sampling rate in Hz
            bands: Frequency band definitions
        """
        self.sample_rate = sample_rate
        self.bands = bands or FrequencyBands()

    def extract_band_power(
        self,
        data: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract power in standard frequency bands.

        Args:
            data: EEG data (samples, channels)
            normalize: Apply log transform and z-score

        Returns:
            Dictionary mapping band names to power arrays (channels,)
        """
        # Compute power spectral density
        freqs, psd = signal.welch(
            data, fs=self.sample_rate,
            nperseg=min(256, len(data)),
            axis=0
        )

        band_power = {}
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']:
            low, high = getattr(self.bands, band_name)

            # Find frequency indices
            idx = np.logical_and(freqs >= low, freqs <= high)

            # Integrate power in band
            power = np.trapz(psd[idx], freqs[idx], axis=0)

            if normalize:
                power = np.log10(power + 1e-10)

            band_power[band_name] = power

        return band_power

    def extract_spectral_entropy(self, data: np.ndarray) -> np.ndarray:
        """
        Compute spectral entropy for each channel.

        Lower entropy = more regular/predictable signal
        Higher entropy = more complex/irregular signal
        """
        freqs, psd = signal.welch(
            data, fs=self.sample_rate,
            nperseg=min(256, len(data)),
            axis=0
        )

        # Normalize PSD to probability distribution
        psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-10)

        # Compute Shannon entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=0)

        return entropy

    def extract_hjorth(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract Hjorth parameters.

        - Activity: Variance of signal (signal power)
        - Mobility: Mean frequency
        - Complexity: Bandwidth
        """
        # First derivative
        diff1 = np.diff(data, axis=0)
        # Second derivative
        diff2 = np.diff(diff1, axis=0)

        # Variances
        var0 = np.var(data, axis=0)
        var1 = np.var(diff1, axis=0)
        var2 = np.var(diff2, axis=0)

        # Hjorth parameters
        activity = var0
        mobility = np.sqrt(var1 / (var0 + 1e-10))
        complexity = np.sqrt(var2 / (var1 + 1e-10)) / (mobility + 1e-10)

        return {
            'activity': activity,
            'mobility': mobility,
            'complexity': complexity
        }

    def extract_statistics(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract statistical features."""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'skew': self._skewness(data),
            'kurtosis': self._kurtosis(data),
            'rms': np.sqrt(np.mean(data**2, axis=0)),
            'peak_to_peak': np.ptp(data, axis=0)
        }

    def _skewness(self, data: np.ndarray) -> np.ndarray:
        """Compute skewness."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-10
        return np.mean(((data - mean) / std) ** 3, axis=0)

    def _kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Compute kurtosis."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-10
        return np.mean(((data - mean) / std) ** 4, axis=0) - 3

    def extract_all(self, data: np.ndarray) -> np.ndarray:
        """
        Extract all features and flatten to feature vector.

        Args:
            data: EEG window (samples, channels)

        Returns:
            Feature vector of shape (n_features,)
        """
        features = []

        # Band power (6 bands * channels)
        band_power = self.extract_band_power(data)
        for band in ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']:
            features.append(band_power[band])

        # Hjorth parameters (3 * channels)
        hjorth = self.extract_hjorth(data)
        features.append(hjorth['activity'])
        features.append(hjorth['mobility'])
        features.append(hjorth['complexity'])

        # Statistics (6 * channels)
        stats = self.extract_statistics(data)
        features.append(stats['mean'])
        features.append(stats['std'])
        features.append(stats['skew'])
        features.append(stats['kurtosis'])
        features.append(stats['rms'])
        features.append(stats['peak_to_peak'])

        # Spectral entropy (1 * channels)
        features.append(self.extract_spectral_entropy(data))

        return np.concatenate(features)

    def get_feature_names(self, channels: int) -> list:
        """Get names for all features."""
        names = []

        # Band power
        for band in ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']:
            for ch in range(channels):
                names.append(f'{band}_ch{ch}')

        # Hjorth
        for param in ['activity', 'mobility', 'complexity']:
            for ch in range(channels):
                names.append(f'hjorth_{param}_ch{ch}')

        # Statistics
        for stat in ['mean', 'std', 'skew', 'kurtosis', 'rms', 'ptp']:
            for ch in range(channels):
                names.append(f'{stat}_ch{ch}')

        # Entropy
        for ch in range(channels):
            names.append(f'spectral_entropy_ch{ch}')

        return names


if __name__ == '__main__':
    # Test feature extraction
    sample_rate = 1000
    channels = 32
    window_samples = 1000  # 1 second

    # Generate test data
    data = np.random.randn(window_samples, channels) * 20  # 20 uV

    extractor = FeatureExtractor(sample_rate=sample_rate)

    # Extract features
    features = extractor.extract_all(data)
    feature_names = extractor.get_feature_names(channels)

    print("Feature Extraction Test")
    print(f"  Input shape: {data.shape}")
    print(f"  Feature vector length: {len(features)}")
    print(f"  Feature names: {len(feature_names)}")

    # Band power
    band_power = extractor.extract_band_power(data)
    print("\nBand Power (channel 0):")
    for band, power in band_power.items():
        print(f"  {band}: {power[0]:.3f}")
