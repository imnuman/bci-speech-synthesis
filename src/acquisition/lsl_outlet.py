"""
Lab Streaming Layer (LSL) Outlet for EEG Data
Publishes EEG stream for time-synchronized multi-modal recording
"""

import numpy as np
from typing import Optional, List
import time


class LSLOutlet:
    """
    LSL outlet for publishing EEG data stream.

    Lab Streaming Layer (LSL) is the standard protocol for BCI systems.
    It handles time synchronization between different data sources
    (EEG, audio, markers) automatically.

    Stream format:
        - Name: "BCI_EEG"
        - Type: "EEG"
        - Channels: 32 (configurable)
        - Sample rate: 1000 Hz (configurable)
        - Format: float32
    """

    def __init__(
        self,
        name: str = "BCI_EEG",
        stream_type: str = "EEG",
        channels: int = 32,
        sample_rate: float = 1000.0,
        channel_names: Optional[List[str]] = None
    ):
        """
        Initialize LSL outlet.

        Args:
            name: Stream name
            stream_type: Stream type identifier
            channels: Number of channels
            sample_rate: Nominal sample rate
            channel_names: List of channel labels
        """
        self.name = name
        self.stream_type = stream_type
        self.channels = channels
        self.sample_rate = sample_rate
        self.channel_names = channel_names or self._default_channel_names()

        self._outlet = None
        self._info = None

    def _default_channel_names(self) -> List[str]:
        """Generate default 10-20 system channel names."""
        # Standard 32-channel layout
        names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'CP5', 'CP1', 'CP2', 'CP6',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'O1', 'O2'
        ]
        return names[:self.channels]

    def open(self) -> bool:
        """Create and open LSL outlet."""
        try:
            from pylsl import StreamInfo, StreamOutlet, cf_float32

            # Create stream info
            self._info = StreamInfo(
                name=self.name,
                type=self.stream_type,
                channel_count=self.channels,
                nominal_srate=self.sample_rate,
                channel_format=cf_float32,
                source_id=f'bci_eeg_{int(time.time())}'
            )

            # Add channel metadata
            channels = self._info.desc().append_child("channels")
            for name in self.channel_names:
                ch = channels.append_child("channel")
                ch.append_child_value("label", name)
                ch.append_child_value("unit", "microvolts")
                ch.append_child_value("type", "EEG")

            # Add acquisition metadata
            acq = self._info.desc().append_child("acquisition")
            acq.append_child_value("manufacturer", "JNEEG")
            acq.append_child_value("model", "32-Channel SPI Shield")
            acq.append_child_value("precision", "24-bit")

            # Create outlet
            self._outlet = StreamOutlet(self._info)

            print(f"LSL outlet '{self.name}' created")
            print(f"  Channels: {self.channels}")
            print(f"  Sample rate: {self.sample_rate} Hz")

            return True

        except ImportError:
            print("pylsl not installed. Install with: pip install pylsl")
            return False
        except Exception as e:
            print(f"Failed to create LSL outlet: {e}")
            return False

    def push_sample(self, sample: np.ndarray, timestamp: Optional[float] = None):
        """
        Push a single sample to the outlet.

        Args:
            sample: Array of shape (channels,)
            timestamp: Optional LSL timestamp (uses current time if None)
        """
        if self._outlet is None:
            return

        if timestamp is not None:
            self._outlet.push_sample(sample.tolist(), timestamp)
        else:
            self._outlet.push_sample(sample.tolist())

    def push_chunk(self, chunk: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """
        Push multiple samples to the outlet.

        Args:
            chunk: Array of shape (n_samples, channels)
            timestamps: Optional array of timestamps
        """
        if self._outlet is None:
            return

        if timestamps is not None:
            self._outlet.push_chunk(chunk.tolist(), timestamps.tolist())
        else:
            self._outlet.push_chunk(chunk.tolist())

    def close(self):
        """Close the LSL outlet."""
        self._outlet = None
        self._info = None


class LSLMarkerOutlet:
    """LSL outlet for event markers."""

    def __init__(self, name: str = "BCI_Markers"):
        """Initialize marker outlet."""
        self.name = name
        self._outlet = None

    def open(self) -> bool:
        """Create marker outlet."""
        try:
            from pylsl import StreamInfo, StreamOutlet, cf_string

            info = StreamInfo(
                name=self.name,
                type="Markers",
                channel_count=1,
                nominal_srate=0,  # Irregular rate
                channel_format=cf_string,
                source_id=f'bci_markers_{int(time.time())}'
            )

            self._outlet = StreamOutlet(info)
            return True

        except Exception as e:
            print(f"Failed to create marker outlet: {e}")
            return False

    def push_marker(self, marker: str, timestamp: Optional[float] = None):
        """Push an event marker."""
        if self._outlet is None:
            return

        if timestamp is not None:
            self._outlet.push_sample([marker], timestamp)
        else:
            self._outlet.push_sample([marker])


if __name__ == '__main__':
    # Test LSL outlet (without actual pylsl)
    outlet = LSLOutlet(channels=32, sample_rate=1000)
    print("LSL outlet module loaded")
    print(f"Channel names: {outlet.channel_names[:8]}...")
