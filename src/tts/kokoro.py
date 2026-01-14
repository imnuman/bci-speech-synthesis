"""
Kokoro-82M Text-to-Speech for BCI Voice Output
Ultra-lightweight TTS with natural voice quality
"""

import numpy as np
from typing import Optional
import time


class KokoroTTS:
    """
    Kokoro-82M Text-to-Speech engine.

    Kokoro is a 2026-standard lightweight TTS model:
    - 82M parameters (vs 1B+ for typical models)
    - <100ms synthesis latency for short phrases
    - Human-grade audio quality
    - ONNX format for cross-platform inference

    Optimized for edge deployment on Jetson Orin NX.
    """

    def __init__(
        self,
        model_path: str = 'models/kokoro-82m.onnx',
        voice: str = 'af_sarah',
        sample_rate: int = 24000,
        device: str = 'cuda'
    ):
        """
        Initialize Kokoro TTS.

        Args:
            model_path: Path to ONNX model
            voice: Voice preset name
            sample_rate: Output sample rate
            device: 'cuda' or 'cpu'
        """
        self.model_path = model_path
        self.voice = voice
        self.sample_rate = sample_rate
        self.device = device

        self._session = None
        self._phonemizer = None

        # Performance tracking
        self._synthesis_times = []

    def load(self) -> bool:
        """Load Kokoro model."""
        try:
            import onnxruntime as ort

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.device == 'cpu':
                providers = ['CPUExecutionProvider']

            self._session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )

            print(f"Loaded Kokoro TTS: {self.model_path}")
            print(f"  Voice: {self.voice}")
            print(f"  Sample rate: {self.sample_rate} Hz")

            return True

        except FileNotFoundError:
            print(f"Model not found: {self.model_path}")
            return False
        except ImportError:
            print("onnxruntime not installed. Install with:")
            print("  pip install onnxruntime-gpu")
            return False
        except Exception as e:
            print(f"Failed to load Kokoro: {e}")
            return False

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Input text to speak

        Returns:
            Audio waveform (float32, -1 to 1)
        """
        start_time = time.perf_counter()

        if self._session is None:
            audio = self._dummy_synthesize(text)
        else:
            audio = self._run_inference(text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._synthesis_times.append(elapsed_ms)

        return audio

    def _run_inference(self, text: str) -> np.ndarray:
        """Run actual ONNX inference."""
        # Tokenize text (simplified)
        tokens = self._text_to_tokens(text)

        # Run inference
        inputs = {
            'tokens': np.array([tokens], dtype=np.int64),
            'voice': np.array([self._voice_embedding()], dtype=np.float32)
        }

        outputs = self._session.run(None, inputs)
        audio = outputs[0][0]

        return audio.astype(np.float32)

    def _text_to_tokens(self, text: str) -> list:
        """Convert text to token IDs (simplified)."""
        # Real implementation would use phonemizer + vocabulary
        return [ord(c) for c in text.lower()[:256]]

    def _voice_embedding(self) -> np.ndarray:
        """Get voice embedding vector."""
        # Real implementation would load from voice file
        return np.zeros(256, dtype=np.float32)

    def _dummy_synthesize(self, text: str) -> np.ndarray:
        """Generate dummy audio when model not loaded."""
        # Simulate synthesis time
        time.sleep(0.05)

        # Generate simple beep for each word
        duration = 0.2 * len(text.split())
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # Simple tone
        freq = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * freq * t)

        # Apply envelope
        attack = int(0.01 * self.sample_rate)
        release = int(0.05 * self.sample_rate)
        envelope = np.ones_like(audio)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)

        return (audio * envelope).astype(np.float32)

    def get_avg_latency(self) -> float:
        """Get average synthesis latency in ms."""
        if not self._synthesis_times:
            return 0.0
        return np.mean(self._synthesis_times[-100:])


class PiperTTS:
    """
    Alternative TTS using Piper (C++ based).

    Even faster than Kokoro for simple responses.
    Good fallback for CPU-only scenarios.
    """

    def __init__(
        self,
        model_path: str = 'models/piper-en-us.onnx',
        sample_rate: int = 22050
    ):
        """Initialize Piper TTS."""
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._model = None

    def load(self) -> bool:
        """Load Piper model."""
        try:
            # Piper uses its own inference engine
            import piper
            self._model = piper.PiperVoice.load(self.model_path)
            return True
        except ImportError:
            print("piper-tts not installed")
            return False
        except Exception as e:
            print(f"Failed to load Piper: {e}")
            return False

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech."""
        if self._model is None:
            # Dummy audio
            return np.zeros(self.sample_rate, dtype=np.float32)

        audio_bytes = b''.join(self._model.synthesize_stream_raw(text))
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768


# Quick responses for common intents
QUICK_RESPONSES = {
    'yes': "Yes",
    'no': "No",
    'help': "I need help",
    'water': "I need water",
    'bathroom': "I need to use the bathroom",
    'pain': "I am in pain",
    'tired': "I am tired",
    'hungry': "I am hungry",
    'cold': "I am cold",
    'hot': "I am hot"
}


def get_response_text(intent: str) -> str:
    """Get natural language response for intent."""
    return QUICK_RESPONSES.get(intent, intent)


if __name__ == '__main__':
    # Test TTS
    tts = KokoroTTS()

    # Test with common intents
    intents = ['yes', 'no', 'water', 'help']

    print("Kokoro TTS Test (dummy mode)")
    for intent in intents:
        text = get_response_text(intent)
        audio = tts.synthesize(text)
        print(f"  '{text}' -> {len(audio)} samples ({len(audio)/tts.sample_rate:.2f}s)")

    print(f"\nAverage latency: {tts.get_avg_latency():.1f}ms")
