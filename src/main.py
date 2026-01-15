"""
BCI Speech Synthesis - Main Entry Point
Real-time brain-to-speech pipeline
"""

import argparse
import yaml
import time
import numpy as np
from typing import Optional
from pathlib import Path

from .acquisition import SPIReader, RingBuffer, LSLOutlet
from .processing import SignalProcessor, FeatureExtractor
from .decoder import TensorRTDecoder
from .context import WhisperSTT, ContextFusion
from .tts import KokoroTTS, ALSAOutput, get_response_text


class BCIPipeline:
    """
    Complete BCI Speech Synthesis Pipeline.

    Data flow:
    EEG Cap -> SPI -> Processing -> Decoder -> Context Fusion -> TTS -> Speaker

    Target latency: <200ms end-to-end
    """

    def __init__(self, config_path: str = 'config/default.yaml'):
        """Initialize pipeline from config file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = self.config['system']['device']

        # Components (initialized in setup())
        self.spi_reader: Optional[SPIReader] = None
        self.buffer: Optional[RingBuffer] = None
        self.lsl_outlet: Optional[LSLOutlet] = None
        self.processor: Optional[SignalProcessor] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.decoder: Optional[TensorRTDecoder] = None
        self.stt: Optional[WhisperSTT] = None
        self.context_fusion: Optional[ContextFusion] = None
        self.tts: Optional[KokoroTTS] = None
        self.audio_output: Optional[ALSAOutput] = None

        # State
        self._running = False
        self._last_intent = None
        self._last_intent_time = 0

    def setup(self) -> bool:
        """Initialize all pipeline components."""
        print("Initializing BCI Pipeline...")

        acq_cfg = self.config['acquisition']
        proc_cfg = self.config['processing']
        dec_cfg = self.config['decoder']
        ctx_cfg = self.config['context']
        tts_cfg = self.config['tts']
        audio_cfg = self.config['audio']

        # 1. EEG Acquisition
        print("  [1/7] SPI Reader...")
        self.spi_reader = SPIReader(
            device=acq_cfg['spi_device'],
            sample_rate=acq_cfg['sample_rate'],
            channels=acq_cfg['channels'],
            speed_hz=acq_cfg['spi_speed_hz']
        )

        # 2. Ring Buffer
        print("  [2/7] Ring Buffer...")
        buffer_samples = acq_cfg['sample_rate'] * acq_cfg['buffer_seconds']
        self.buffer = RingBuffer(
            capacity=buffer_samples,
            channels=acq_cfg['channels']
        )

        # 3. LSL Outlet (optional)
        print("  [3/7] LSL Outlet...")
        self.lsl_outlet = LSLOutlet(
            channels=acq_cfg['channels'],
            sample_rate=acq_cfg['sample_rate']
        )
        if not self.lsl_outlet.open():
            print("    Warning: LSL outlet failed to open")
            self.lsl_outlet = None

        # 4. Signal Processing
        print("  [4/7] Signal Processor...")
        self.processor = SignalProcessor(
            sample_rate=acq_cfg['sample_rate'],
            notch_freq=proc_cfg['notch_freq'],
            bandpass=(proc_cfg['bandpass_low'], proc_cfg['bandpass_high']),
            artifact_threshold=proc_cfg['artifact_threshold'],
            use_car=proc_cfg['use_car']
        )
        self.feature_extractor = FeatureExtractor(
            sample_rate=acq_cfg['sample_rate']
        )

        # 5. Neural Decoder
        print("  [5/7] Decoder...")
        self.decoder = TensorRTDecoder(
            engine_path=dec_cfg['model_path'],
            vocab=dec_cfg['vocab']
        )
        if not self.decoder.load():
            print("    Warning: TensorRT decoder using fallback mode")
        else:
            print("    Warming up decoder...")
            self.decoder.warmup()

        # 6. Context Awareness
        if ctx_cfg['enabled']:
            print("  [6/7] Context (STT + Fusion)...")
            self.stt = WhisperSTT(
                model=ctx_cfg['stt_model'],
                language=ctx_cfg['stt_language'],
                device=ctx_cfg['stt_device']
            )
            if not self.stt.load():
                print("    Warning: Whisper STT using fallback mode")
            self.context_fusion = ContextFusion(
                context_window_sec=ctx_cfg['context_window_sec'],
                boost_factor=ctx_cfg['boost_factor']
            )
        else:
            print("  [6/7] Context disabled")

        # 7. TTS + Audio
        print("  [7/7] TTS + Audio Output...")
        self.tts = KokoroTTS(
            model_path=tts_cfg['model_path'],
            voice=tts_cfg['voice'],
            sample_rate=tts_cfg['sample_rate']
        )
        if not self.tts.load():
            print("    Warning: TTS using fallback mode")
        
        self.audio_output = ALSAOutput(
            device=audio_cfg['device'],
            sample_rate=audio_cfg['sample_rate'],
            buffer_size=audio_cfg['buffer_size']
        )
        if not self.audio_output.open():
            print("    Warning: Audio output using fallback mode")

        print("Pipeline initialized!")
        return True

    def run(self):
        """Run the main processing loop."""
        print("\nStarting BCI Pipeline...")
        print("Press Ctrl+C to stop\n")

        dec_cfg = self.config['decoder']
        window_size = dec_cfg['window_size']
        hop_size = dec_cfg['hop_size']
        confidence_threshold = dec_cfg['confidence_threshold']

        self._running = True
        samples_since_process = 0

        # Callback for SPI data
        def on_sample(sample):
            nonlocal samples_since_process
            self.buffer.push(sample)
            samples_since_process += 1

            # Publish to LSL
            if self.lsl_outlet:
                self.lsl_outlet.push_sample(sample)

        # Start acquisition
        self.spi_reader.start(callback=on_sample)

        try:
            while self._running:
                # Wait for enough new samples
                if samples_since_process < hop_size:
                    time.sleep(0.01)
                    continue

                samples_since_process = 0

                # Get processing window
                window = self.buffer.get_window(window_size)
                if window is None:
                    continue

                # Process EEG
                start_time = time.perf_counter()

                processed, is_clean = self.processor.process_window(window)
                if not is_clean:
                    continue

                # Extract features
                features = self.feature_extractor.extract_all(processed)

                # Decode intent
                intent, confidence, probs = self.decoder.predict(
                    features.reshape(1, -1)
                )

                # Apply context if available
                if self.context_fusion:
                    intent = self.context_fusion.refine(probs)

                # Check confidence threshold
                if confidence < confidence_threshold:
                    continue

                # Debounce (don't repeat same intent within 1 second)
                current_time = time.time()
                if intent == self._last_intent and (current_time - self._last_intent_time) < 1.0:
                    continue

                # Output!
                self._last_intent = intent
                self._last_intent_time = current_time

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Synthesize and play
                text = get_response_text(intent)
                print(f"  [{latency_ms:.0f}ms] {intent} ({confidence:.1%}): \"{text}\"")

                audio = self.tts.synthesize(text)
                self.audio_output.play(audio)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.spi_reader.stop()
            self._running = False

    def demo_mode(self):
        """Run in demo mode with simulated EEG."""
        print("\nRunning in DEMO mode (simulated EEG)...")
        print("Press Ctrl+C to stop\n")

        dec_cfg = self.config['decoder']
        vocab = dec_cfg['vocab']

        self._running = True

        try:
            while self._running:
                # Simulate EEG processing delay
                time.sleep(0.5)

                # Random intent with varying confidence
                probs = np.random.dirichlet(np.ones(len(vocab)) * 0.5)
                idx = np.argmax(probs)
                intent = vocab[idx]
                confidence = probs[idx]

                if confidence < 0.3:
                    print(f"  [low confidence] {intent} ({confidence:.1%})")
                    continue

                text = get_response_text(intent)
                print(f"  [DEMO] {intent} ({confidence:.1%}): \"{text}\"")

                # Synthesize (dummy)
                audio = self.tts.synthesize(text)
                print(f"         Audio: {len(audio)} samples")

        except KeyboardInterrupt:
            print("\nStopping...")

        self._running = False


def main():
    parser = argparse.ArgumentParser(description='BCI Speech Synthesis')
    parser.add_argument(
        '--config', '-c',
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode with simulated EEG'
    )

    args = parser.parse_args()

    pipeline = BCIPipeline(config_path=args.config)
    pipeline.setup()

    if args.demo:
        pipeline.demo_mode()
    else:
        pipeline.run()


if __name__ == '__main__':
    main()
