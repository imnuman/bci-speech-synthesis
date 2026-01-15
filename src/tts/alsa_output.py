"""
ALSA Audio Output for Low-Latency Speech Playback
Direct hardware access for minimal audio latency
"""

import numpy as np
from typing import Optional
import threading
import queue


class ALSAOutput:
    """
    Low-latency audio output using ALSA.

    ALSA (Advanced Linux Sound Architecture) provides the lowest
    latency audio output on Linux. We avoid PulseAudio/PipeWire
    to save ~10-20ms of latency.

    Configuration for Jetson:
    - Device: 'plughw:0,0' or 'default'
    - Buffer size: 1024 samples (43ms @ 24kHz)
    - Period size: 512 samples (21ms @ 24kHz)
    """

    def __init__(
        self,
        device: str = 'default',
        sample_rate: int = 24000,
        channels: int = 1,
        buffer_size: int = 1024
    ):
        """
        Initialize ALSA output.

        Args:
            device: ALSA device name
            sample_rate: Audio sample rate
            channels: Number of channels (1 for mono)
            buffer_size: Buffer size in samples
        """
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size

        self._pcm = None
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def open(self) -> bool:
        """Open ALSA device."""
        try:
            import alsaaudio

            self._pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_PLAYBACK,
                mode=alsaaudio.PCM_NORMAL,
                device=self.device
            )

            self._pcm.setchannels(self.channels)
            self._pcm.setrate(self.sample_rate)
            self._pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            self._pcm.setperiodsize(self.buffer_size)

            print(f"Opened ALSA device: {self.device}")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Buffer size: {self.buffer_size} samples")
            print(f"  Latency: {self.buffer_size / self.sample_rate * 1000:.1f}ms")

            return True

        except ImportError:
            print("pyalsaaudio not installed. Install with:")
            print("  apt-get install libasound2-dev")
            print("  pip install pyalsaaudio")
            return False
        except Exception as e:
            print(f"Failed to open ALSA device: {e}")
            return False

    def play(self, audio: np.ndarray, block: bool = True):
        """
        Play audio data.

        Args:
            audio: Audio waveform (float32, -1 to 1)
            block: Wait for playback to complete
        """
        if self._pcm is None:
            self._dummy_play(audio)
            return

        # Clip and convert to 16-bit PCM
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # Write to device
        self._pcm.write(audio_int16.tobytes())

    def play_async(self, audio: np.ndarray):
        """Queue audio for async playback."""
        self._queue.put(audio)

        if not self._running:
            self._start_playback_thread()

    def _start_playback_thread(self):
        """Start background playback thread."""
        self._running = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def _playback_loop(self):
        """Background playback loop."""
        while self._running:
            try:
                audio = self._queue.get(timeout=0.1)
                self.play(audio)
            except queue.Empty:
                continue

    def _dummy_play(self, audio: np.ndarray):
        """Simulate playback when ALSA not available."""
        import time
        duration = len(audio) / self.sample_rate
        time.sleep(duration)

    def stop(self):
        """Stop playback."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def close(self):
        """Close ALSA device."""
        self.stop()
        if self._pcm:
            self._pcm.close()
            self._pcm = None


class SoundDeviceOutput:
    """
    Alternative audio output using sounddevice.

    Cross-platform fallback when ALSA not available.
    Uses PortAudio backend.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1
    ):
        """Initialize sounddevice output."""
        self.sample_rate = sample_rate
        self.channels = channels

    def play(self, audio: np.ndarray, block: bool = True):
        """Play audio using sounddevice."""
        try:
            import sounddevice as sd
            sd.play(audio, self.sample_rate, blocking=block)
        except ImportError:
            print("sounddevice not installed")
        except Exception as e:
            print(f"Playback error: {e}")

    def stop(self):
        """Stop playback."""
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass


if __name__ == '__main__':
    # Test audio output
    output = ALSAOutput(sample_rate=24000)

    # Generate test tone
    duration = 0.5
    t = np.linspace(0, duration, int(24000 * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("ALSA Output Test")
    print(f"  Audio duration: {duration}s")
    print(f"  Audio samples: {len(audio)}")

    # Try to play (will use dummy if ALSA not available)
    output.play(audio)
    print("  Playback complete")
