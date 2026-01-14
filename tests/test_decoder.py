"""Tests for neural decoder module"""

import pytest
import numpy as np
import torch
import sys
sys.path.insert(0, 'src')

from decoder.tcn import TemporalConvNet, TemporalBlock, CausalConv1d, SpeechIntentDecoder


class TestCausalConv1d:
    """Tests for CausalConv1d layer."""

    def test_output_shape(self):
        """Test output matches input length (causal)."""
        conv = CausalConv1d(32, 64, kernel_size=3, dilation=1)

        x = torch.randn(1, 32, 100)
        y = conv(x)

        assert y.shape == (1, 64, 100)

    def test_dilated_output_shape(self):
        """Test dilated convolution maintains length."""
        conv = CausalConv1d(32, 64, kernel_size=3, dilation=4)

        x = torch.randn(1, 32, 100)
        y = conv(x)

        assert y.shape == (1, 64, 100)


class TestTemporalBlock:
    """Tests for TemporalBlock class."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        block = TemporalBlock(32, 64, kernel_size=3, dilation=1)

        x = torch.randn(1, 32, 100)
        y = block(x)

        assert y.shape == (1, 64, 100)

    def test_residual_connection(self):
        """Test residual connection works."""
        # Same channels - identity residual
        block = TemporalBlock(32, 32, kernel_size=3, dilation=1)
        assert isinstance(block.residual, torch.nn.Identity)

        # Different channels - 1x1 conv residual
        block = TemporalBlock(32, 64, kernel_size=3, dilation=1)
        assert isinstance(block.residual, torch.nn.Conv1d)


class TestTemporalConvNet:
    """Tests for TemporalConvNet model."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return TemporalConvNet(
            input_channels=32,
            num_classes=10,
            hidden_channels=[32, 32, 64],
            kernel_size=3,
            dropout=0.0
        )

    def test_forward_pass(self, model):
        """Test forward pass produces correct shape."""
        x = torch.randn(4, 32, 1000)  # batch=4, channels=32, time=1000
        y = model(x)

        assert y.shape == (4, 10)

    def test_predict(self, model):
        """Test predict returns probabilities."""
        x = torch.randn(1, 32, 1000)
        probs = model.predict(x)

        assert probs.shape == (1, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_receptive_field(self, model):
        """Test receptive field calculation."""
        rf = model.get_receptive_field()
        assert rf > 0
        assert isinstance(rf, int)

    def test_batch_processing(self, model):
        """Test batch processing works correctly."""
        batch_sizes = [1, 2, 4, 8]

        for bs in batch_sizes:
            x = torch.randn(bs, 32, 1000)
            y = model(x)
            assert y.shape == (bs, 10)

    def test_variable_sequence_length(self, model):
        """Test different sequence lengths."""
        lengths = [500, 1000, 2000]

        for length in lengths:
            x = torch.randn(1, 32, length)
            y = model(x)
            assert y.shape == (1, 10)


class TestSpeechIntentDecoder:
    """Tests for SpeechIntentDecoder class."""

    @pytest.fixture
    def decoder(self):
        """Create decoder for testing."""
        return SpeechIntentDecoder(
            input_channels=32,
            window_size=1000,
            device='cpu'
        )

    def test_initialization(self, decoder):
        """Test decoder initialization."""
        assert len(decoder.vocab) == 10
        assert decoder.window_size == 1000

    def test_decode(self, decoder):
        """Test decoding returns valid intent."""
        eeg = torch.randn(32, 1000)
        intent, confidence, probs = decoder.decode(eeg)

        assert intent in decoder.vocab
        assert 0 <= confidence <= 1
        assert len(probs) == len(decoder.vocab)

    def test_decode_batch(self, decoder):
        """Test decoding with batch dimension."""
        eeg = torch.randn(1, 32, 1000)
        intent, confidence, probs = decoder.decode(eeg)

        assert intent in decoder.vocab

    def test_custom_vocab(self):
        """Test decoder with custom vocabulary."""
        vocab = ['up', 'down', 'left', 'right']
        decoder = SpeechIntentDecoder(
            input_channels=32,
            vocab=vocab,
            device='cpu'
        )

        assert decoder.vocab == vocab

        eeg = torch.randn(32, 1000)
        intent, _, _ = decoder.decode(eeg)
        assert intent in vocab


class TestModelExport:
    """Tests for model export functionality."""

    def test_onnx_export(self, tmp_path):
        """Test ONNX export creates valid file."""
        decoder = SpeechIntentDecoder(
            input_channels=32,
            window_size=500,
            device='cpu'
        )

        onnx_path = tmp_path / "model.onnx"
        decoder.export_onnx(str(onnx_path))

        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
