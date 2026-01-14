"""
Real-time Signal Processing Filters for EEG
Optimized for low-latency streaming data
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FilterState:
    """State container for IIR filter (for real-time streaming)."""
    zi: np.ndarray  # Filter initial conditions


class NotchFilter:
    """
    Real-time notch filter for power line interference removal.

    Removes 50Hz (Bangladesh/Europe) or 60Hz (Americas) interference
    and harmonics from EEG signal.
    """

    def __init__(
        self,
        sample_rate: float,
        freq: float = 50.0,
        quality: float = 30.0,
        harmonics: int = 3
    ):
        """
        Initialize notch filter.

        Args:
            sample_rate: Sampling rate in Hz
            freq: Notch frequency (50 or 60 Hz)
            quality: Q factor (higher = narrower notch)
            harmonics: Number of harmonics to filter
        """
        self.sample_rate = sample_rate
        self.freq = freq
        self.quality = quality

        # Design notch filters for fundamental and harmonics
        self.filters = []
        for h in range(1, harmonics + 1):
            notch_freq = freq * h
            if notch_freq < sample_rate / 2:  # Below Nyquist
                b, a = signal.iirnotch(notch_freq, quality, sample_rate)
                self.filters.append((b, a))

        self._states: Optional[list] = None

    def initialize(self, channels: int):
        """Initialize filter states for streaming."""
        self._states = []
        for b, a in self.filters:
            zi = signal.lfilter_zi(b, a)
            # Expand for all channels
            self._states.append(np.tile(zi, (channels, 1)).T)

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to data.

        Args:
            data: Array of shape (samples, channels) or (channels, samples)

        Returns:
            Filtered data with same shape
        """
        # Ensure (samples, channels) format
        if data.shape[0] < data.shape[1]:
            data = data.T
            transpose_back = True
        else:
            transpose_back = False

        # Initialize states if needed
        if self._states is None:
            self.initialize(data.shape[1])

        # Apply cascade of notch filters
        filtered = data.copy()
        for i, (b, a) in enumerate(self.filters):
            filtered, self._states[i] = signal.lfilter(
                b, a, filtered, axis=0, zi=self._states[i]
            )

        if transpose_back:
            filtered = filtered.T

        return filtered


class BandpassFilter:
    """
    Real-time bandpass filter for EEG frequency band selection.

    Standard EEG bands:
    - Delta: 0.5-4 Hz (deep sleep)
    - Theta: 4-8 Hz (drowsiness, meditation)
    - Alpha: 8-13 Hz (relaxed, eyes closed)
    - Beta: 13-30 Hz (active thinking)
    - Gamma: 30-100 Hz (high-level cognition, speech)
    """

    def __init__(
        self,
        sample_rate: float,
        low_freq: float = 1.0,
        high_freq: float = 50.0,
        order: int = 4
    ):
        """
        Initialize bandpass filter.

        Args:
            sample_rate: Sampling rate in Hz
            low_freq: Lower cutoff frequency
            high_freq: Upper cutoff frequency
            order: Filter order
        """
        self.sample_rate = sample_rate

        # Design Butterworth bandpass filter
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = min(high_freq / nyquist, 0.99)

        self.b, self.a = signal.butter(order, [low, high], btype='band')
        self._state: Optional[np.ndarray] = None

    def initialize(self, channels: int):
        """Initialize filter state for streaming."""
        zi = signal.lfilter_zi(self.b, self.a)
        self._state = np.tile(zi, (channels, 1)).T

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to data."""
        if data.shape[0] < data.shape[1]:
            data = data.T
            transpose_back = True
        else:
            transpose_back = False

        if self._state is None:
            self.initialize(data.shape[1])

        filtered, self._state = signal.lfilter(
            self.b, self.a, data, axis=0, zi=self._state
        )

        if transpose_back:
            filtered = filtered.T

        return filtered


class SignalProcessor:
    """
    Complete real-time signal processing pipeline for EEG.

    Pipeline:
    1. Notch filter (remove power line noise)
    2. Bandpass filter (extract frequency band of interest)
    3. Artifact rejection (threshold-based)
    4. Common average reference (optional)
    """

    def __init__(
        self,
        sample_rate: float = 1000.0,
        notch_freq: float = 50.0,
        bandpass: Tuple[float, float] = (1.0, 50.0),
        artifact_threshold: float = 100.0,
        use_car: bool = True
    ):
        """
        Initialize signal processor.

        Args:
            sample_rate: Sampling rate in Hz
            notch_freq: Power line frequency (50 or 60 Hz)
            bandpass: (low, high) frequency range
            artifact_threshold: Amplitude threshold for artifact rejection (uV)
            use_car: Apply Common Average Reference
        """
        self.sample_rate = sample_rate
        self.artifact_threshold = artifact_threshold
        self.use_car = use_car

        # Initialize filters
        self.notch = NotchFilter(sample_rate, notch_freq)
        self.bandpass = BandpassFilter(sample_rate, bandpass[0], bandpass[1])

        self._initialized = False

    def initialize(self, channels: int):
        """Initialize processor for given number of channels."""
        self.notch.initialize(channels)
        self.bandpass.initialize(channels)
        self._initialized = True

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process EEG data through the pipeline.

        Args:
            data: Raw EEG data (samples, channels)

        Returns:
            Processed EEG data
        """
        if not self._initialized:
            self.initialize(data.shape[1] if data.ndim > 1 else 1)

        # 1. Notch filter
        filtered = self.notch.process(data)

        # 2. Bandpass filter
        filtered = self.bandpass.process(filtered)

        # 3. Common Average Reference
        if self.use_car:
            mean = np.mean(filtered, axis=1, keepdims=True)
            filtered = filtered - mean

        # 4. Artifact rejection (mark as NaN)
        if self.artifact_threshold > 0:
            mask = np.abs(filtered) > self.artifact_threshold
            filtered[mask] = np.nan

        return filtered

    def process_window(self, window: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a window and check for artifacts.

        Args:
            window: EEG window (samples, channels)

        Returns:
            Tuple of (processed_data, is_clean)
        """
        processed = self.process(window)
        is_clean = not np.any(np.isnan(processed))

        if not is_clean:
            # Interpolate NaN values
            processed = np.nan_to_num(processed, nan=0.0)

        return processed, is_clean


if __name__ == '__main__':
    # Test signal processing pipeline
    sample_rate = 1000
    duration = 2.0
    channels = 32

    # Generate test signal with 50Hz noise
    t = np.linspace(0, duration, int(sample_rate * duration))
    clean_signal = np.random.randn(len(t), channels) * 10  # 10 uV noise floor
    noise_50hz = 50 * np.sin(2 * np.pi * 50 * t)[:, np.newaxis]  # 50 uV 50Hz
    raw_signal = clean_signal + noise_50hz

    # Process
    processor = SignalProcessor(
        sample_rate=sample_rate,
        notch_freq=50,
        bandpass=(1, 50)
    )

    filtered = processor.process(raw_signal)

    print("Signal Processing Test")
    print(f"  Input shape: {raw_signal.shape}")
    print(f"  Output shape: {filtered.shape}")
    print(f"  Raw RMS: {np.sqrt(np.mean(raw_signal**2)):.2f} uV")
    print(f"  Filtered RMS: {np.sqrt(np.nanmean(filtered**2)):.2f} uV")
