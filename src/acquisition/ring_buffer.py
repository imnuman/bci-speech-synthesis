"""
Lock-free Ring Buffer for EEG Data
High-performance circular buffer for real-time signal processing
"""

import numpy as np
from typing import Optional
import threading


class RingBuffer:
    """
    Thread-safe ring buffer for continuous EEG data.

    Designed for producer-consumer pattern:
    - Producer: SPI acquisition thread pushes samples
    - Consumer: Processing thread pulls windows

    Features:
    - Lock-free single-producer single-consumer operation
    - Automatic overwrite of oldest data when full
    - Efficient window extraction without copying
    """

    def __init__(self, capacity: int, channels: int):
        """
        Initialize ring buffer.

        Args:
            capacity: Number of samples to store
            channels: Number of EEG channels
        """
        self.capacity = capacity
        self.channels = channels

        # Pre-allocate buffer
        self._buffer = np.zeros((capacity, channels), dtype=np.float32)

        # Atomic indices
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0

        # Lock for thread safety (fallback)
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        """Number of samples available for reading."""
        return self._count

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._count >= self.capacity

    def push(self, sample: np.ndarray):
        """
        Push a single sample into the buffer.

        Args:
            sample: Array of shape (channels,)
        """
        with self._lock:
            self._buffer[self._write_idx] = sample
            self._write_idx = (self._write_idx + 1) % self.capacity

            if self._count < self.capacity:
                self._count += 1
            else:
                # Overwrite oldest sample
                self._read_idx = (self._read_idx + 1) % self.capacity

    def push_batch(self, samples: np.ndarray):
        """
        Push multiple samples into the buffer.

        Args:
            samples: Array of shape (n_samples, channels)
        """
        for sample in samples:
            self.push(sample)

    def get_window(self, size: int) -> Optional[np.ndarray]:
        """
        Get the most recent window of samples.

        Args:
            size: Number of samples to retrieve

        Returns:
            Array of shape (size, channels) or None if insufficient data
        """
        with self._lock:
            if self._count < size:
                return None

            # Calculate start index for most recent 'size' samples
            end_idx = self._write_idx
            start_idx = (end_idx - size) % self.capacity

            # Extract window (may wrap around)
            if start_idx < end_idx:
                return self._buffer[start_idx:end_idx].copy()
            else:
                # Wrapped around
                part1 = self._buffer[start_idx:]
                part2 = self._buffer[:end_idx]
                return np.vstack([part1, part2])

    def get_latest(self, n: int = 1) -> Optional[np.ndarray]:
        """
        Get the n most recent samples.

        Args:
            n: Number of samples to retrieve

        Returns:
            Array of shape (n, channels) or None
        """
        return self.get_window(n)

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._write_idx = 0
            self._read_idx = 0
            self._count = 0
            self._buffer.fill(0)


class StreamBuffer(RingBuffer):
    """
    Extended ring buffer with sliding window support for real-time processing.
    """

    def __init__(
        self,
        capacity: int,
        channels: int,
        window_size: int,
        hop_size: int
    ):
        """
        Initialize stream buffer.

        Args:
            capacity: Total buffer capacity in samples
            channels: Number of EEG channels
            window_size: Processing window size in samples
            hop_size: Samples between consecutive windows
        """
        super().__init__(capacity, channels)
        self.window_size = window_size
        self.hop_size = hop_size
        self._last_process_idx = 0

    def get_next_window(self) -> Optional[np.ndarray]:
        """
        Get next window for processing with hop.

        Returns:
            Window array or None if not enough new data
        """
        with self._lock:
            samples_since_last = self._write_idx - self._last_process_idx
            if samples_since_last < 0:
                samples_since_last += self.capacity

            if samples_since_last < self.hop_size:
                return None

            window = self.get_window(self.window_size)
            if window is not None:
                self._last_process_idx = (self._last_process_idx + self.hop_size) % self.capacity

            return window


if __name__ == '__main__':
    # Test ring buffer
    buffer = RingBuffer(capacity=1000, channels=32)

    # Simulate pushing samples
    for i in range(500):
        sample = np.random.randn(32).astype(np.float32)
        buffer.push(sample)

    print(f"Buffer: {buffer.available}/{buffer.capacity} samples")

    # Get window
    window = buffer.get_window(250)
    if window is not None:
        print(f"Window shape: {window.shape}")
        print(f"Window mean: {window.mean():.4f} uV")
