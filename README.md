# BCI Speech Synthesis

Real-time Brain-Computer Interface for speech synthesis. Converts EEG signals to natural speech with contextual awareness.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![Platform](https://img.shields.io/badge/platform-Jetson_Orin_NX-76B900.svg)
![Status](https://img.shields.io/badge/status-development-yellow.svg)

## Overview

A wearable BCI system that decodes neural signals into spoken words, designed for individuals with speech impairments. The system uses high-density EEG, contextual audio awareness, and low-latency text-to-speech to enable real-time communication.

**Target Latency:** <200ms end-to-end (brain signal → spoken word)

### Key Features

- **32-Channel EEG** via SPI (24-bit, up to 16kHz sampling)
- **Real-time Decoding** using TensorRT-optimized TCN/Transformer
- **Contextual Awareness** - ambient speech recognition narrows decoder output
- **Natural Voice** - Kokoro-82M TTS with <100ms synthesis latency
- **Portable** - Battery-powered, vest-mounted design

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EDGE DEVICE (Jetson Orin NX)                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     EEG ACQUISITION                              │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │    │
│  │   │ Greentek Cap │───▶│ JNEEG Shield │───▶│ /dev/spidev  │     │    │
│  │   │ 32-ch Saline │    │ 4x ADS1299   │    │  0.0         │     │    │
│  │   └──────────────┘    └──────────────┘    └──────┬───────┘     │    │
│  └──────────────────────────────────────────────────┼──────────────┘    │
│                                                      │                   │
│  ┌──────────────────────────────────────────────────▼──────────────┐    │
│  │                   SIGNAL PROCESSING                              │    │
│  │   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────────────┐    │    │
│  │   │ Notch  │──▶│Bandpass│──▶│Artifact│──▶│ Feature Extract│    │    │
│  │   │ 50/60Hz│   │ 1-50Hz │   │Removal │   │ (Band Power)   │    │    │
│  │   └────────┘   └────────┘   └────────┘   └───────┬────────┘    │    │
│  └──────────────────────────────────────────────────┼──────────────┘    │
│                                                      │                   │
│  ┌──────────────────────────────────────────────────▼──────────────┐    │
│  │                    NEURAL DECODER                                │    │
│  │   ┌─────────────────────────────────────────────────────────┐   │    │
│  │   │           TensorRT Engine (TCN / Transformer)            │   │    │
│  │   │              EEG Features → Word Probabilities            │   │    │
│  │   └─────────────────────────┬───────────────────────────────┘   │    │
│  └──────────────────────────────┼──────────────────────────────────┘    │
│                                 │                                        │
│         ┌───────────────────────┴───────────────────────┐               │
│         │                                               │               │
│         ▼                                               ▼               │
│  ┌──────────────┐                              ┌──────────────┐         │
│  │   CONTEXT    │                              │     TTS      │         │
│  │ ┌──────────┐ │      ┌──────────────┐       │ ┌──────────┐ │         │
│  │ │DJI Mic   │─┼─────▶│Context Fusion│◀──────┼─│Kokoro-82M│ │         │
│  │ │+ Whisper │ │      │   (LLM)      │       │ │  ONNX    │ │         │
│  │ └──────────┘ │      └──────┬───────┘       │ └────┬─────┘ │         │
│  └──────────────┘             │               └──────┼───────┘         │
│                               │                      │                  │
│                               ▼                      ▼                  │
│                        ┌─────────────┐        ┌─────────────┐          │
│                        │  Decoded    │        │   ALSA      │          │
│                        │   Intent    │───────▶│   Speaker   │          │
│                        └─────────────┘        └─────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

| Component | Model | Purpose | Cost |
|-----------|-------|---------|------|
| Compute | Jetson Orin NX 16GB | Edge AI processing (100 TOPS) | $1,100 |
| Amplifier | JNEEG 32-ch SPI Shield | 4x ADS1299, 24-bit ADC, up to 16kHz | $700 |
| Sensors | Greentek Gelfree-S3 | 32-ch saline electrode cap | $650 |
| Audio In | DJI Mic Mini + Isolator | Ambient speech capture | $120 |
| Power | 12V 10Ah Li-ion + DC-DC | Portable power supply | $150 |
| **Total** | | | **~$2,720** |

## Software Stack

| Layer | Component | Purpose |
|-------|-----------|---------|
| OS | JetPack 6.x (Ubuntu 22.04) | NVIDIA drivers, CUDA |
| Kernel | PREEMPT_RT patch | Real-time scheduling |
| Driver | spidev | SPI communication |
| Middleware | Lab Streaming Layer (LSL) | Time-synchronized data streams |
| Processing | SciPy, Numba | Real-time signal filtering |
| Decoder | TensorRT (FP16) | Optimized neural inference |
| STT | Faster-Whisper (Int8) | Ambient speech recognition |
| TTS | Kokoro-82M (ONNX) | Natural voice synthesis |
| Audio | ALSA | Low-latency audio output |

## Installation

### Prerequisites

- NVIDIA Jetson Orin NX with JetPack 6.x
- JNEEG 32-Channel SPI Shield connected to GPIO header
- SPI enabled via `jetson-io.py`

### Setup

```bash
# Clone repository
git clone https://github.com/imnuman/bci-speech-synthesis.git
cd bci-speech-synthesis

# Create environment
conda create -n bci python=3.10
conda activate bci

# Install dependencies
pip install -r requirements.txt

# Enable SPI interface (Jetson)
sudo /opt/nvidia/jetson-io/jetson-io.py

# Install LSL
pip install pylsl

# Build TensorRT engine
python scripts/build_tensorrt_engine.py --onnx models/decoder.onnx
```

## Project Structure

```
bci-speech-synthesis/
├── src/
│   ├── acquisition/
│   │   ├── spi_reader.py       # SPI data acquisition
│   │   ├── lsl_outlet.py       # LSL stream publisher
│   │   └── ring_buffer.py      # Lock-free ring buffer
│   ├── processing/
│   │   ├── filters.py          # Bandpass, notch filters
│   │   ├── artifact.py         # Artifact rejection
│   │   └── features.py         # Feature extraction
│   ├── decoder/
│   │   ├── tcn.py              # Temporal Convolutional Network
│   │   ├── transformer.py      # EEG Transformer
│   │   └── tensorrt_infer.py   # TensorRT inference
│   ├── context/
│   │   ├── whisper_stt.py      # Faster-Whisper integration
│   │   └── context_fusion.py   # LLM context refinement
│   └── tts/
│       ├── kokoro.py           # Kokoro-82M TTS
│       └── alsa_output.py      # ALSA audio playback
├── config/
│   ├── spi_config.yaml         # SPI bus configuration
│   ├── processing.yaml         # Signal processing params
│   └── decoder.yaml            # Model configuration
├── models/
│   ├── decoder.onnx            # ONNX decoder model
│   ├── decoder.engine          # TensorRT engine (generated)
│   └── kokoro-82m.onnx         # TTS model
├── scripts/
│   ├── setup_spi.sh            # SPI interface setup
│   ├── build_tensorrt_engine.py
│   └── calibrate_electrodes.py
├── tests/
│   ├── test_acquisition.py
│   ├── test_processing.py
│   └── test_decoder.py
├── docs/
│   ├── HARDWARE_SETUP.md
│   ├── CALIBRATION.md
│   └── TRAINING.md
├── requirements.txt
└── README.md
```

## Usage

### Quick Start

```bash
# Run the full pipeline
python -m src.main --config config/default.yaml
```

### Data Acquisition

```python
from src.acquisition import SPIReader, RingBuffer

# Initialize SPI reader
reader = SPIReader(
    device='/dev/spidev0.0',
    sample_rate=1000,
    channels=32
)

# Create ring buffer (2 seconds)
buffer = RingBuffer(capacity=2000, channels=32)

# Start acquisition
reader.start(callback=buffer.push)
```

### Signal Processing

```python
from src.processing import SignalProcessor

processor = SignalProcessor(
    sample_rate=1000,
    notch_freq=50,          # Power line frequency
    bandpass=(1, 50),       # EEG frequency range
    artifact_threshold=100   # μV
)

# Process EEG window
clean_data = processor.process(raw_eeg)
features = processor.extract_features(clean_data)
```

### Neural Decoding

```python
from src.decoder import TensorRTDecoder

decoder = TensorRTDecoder(
    engine_path='models/decoder.engine',
    vocab=['yes', 'no', 'help', 'water', 'bathroom', 'pain', 'tired']
)

# Decode intent
intent, confidence = decoder.predict(features)
print(f"Intent: {intent} ({confidence:.2%})")
```

### Context-Aware Refinement

```python
from src.context import WhisperSTT, ContextFusion

stt = WhisperSTT(model='small', device='cuda')
fusion = ContextFusion()

# Get ambient speech
ambient_text = stt.transcribe(audio_chunk)  # "Are you thirsty?"

# Refine decoder output with context
refined_intent = fusion.refine(
    decoder_output={'yes': 0.6, 'no': 0.3, 'water': 0.1},
    context=ambient_text
)
# Returns: 'yes' (boosted by question context)
```

### Text-to-Speech

```python
from src.tts import KokoroTTS

tts = KokoroTTS(model_path='models/kokoro-82m.onnx')

# Synthesize speech
audio = tts.synthesize("Yes, I am thirsty")
tts.play(audio)  # ALSA output
```

## Performance Targets

| Stage | Target Latency | Status |
|-------|---------------|--------|
| EEG Acquisition | <10ms | In Development |
| Signal Processing | <20ms | In Development |
| Neural Decoding | <50ms | In Development |
| Context Fusion | <50ms | In Development |
| TTS Synthesis | <70ms | In Development |
| **Total** | **<200ms** | |

## Development Roadmap

- [x] Hardware BOM and architecture design
- [x] Repository structure and scaffolding
- [ ] SPI data acquisition daemon
- [ ] Real-time signal processing pipeline
- [ ] TCN decoder model training
- [ ] TensorRT optimization
- [ ] Faster-Whisper integration
- [ ] Kokoro-82M TTS integration
- [ ] End-to-end system testing
- [ ] Wearable enclosure design

## References

- [ADS1299 Datasheet](https://www.ti.com/product/ADS1299) - TI 24-bit ADC
- [Lab Streaming Layer](https://labstreaminglayer.org/) - Time-synchronized streaming
- [EEGNet](https://arxiv.org/abs/1611.08024) - Compact CNN for BCI
- [Kokoro TTS](https://github.com/hexgrad/kokoro) - Lightweight TTS model

## License

MIT License

## Author

Al Numan - ProjectX

---

*Giving voice to those who cannot speak*
