"""
Temporal Convolutional Network for EEG Decoding
Designed for real-time speech intent classification from brain signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CausalConv1d(nn.Module):
    """Causal convolution - no future information leakage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        res = self.residual(x)
        return F.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for EEG-based speech decoding.

    Optimized for real-time inference on Jetson Orin NX with TensorRT.
    """

    def __init__(
        self,
        input_channels: int = 32,
        num_classes: int = 10,
        hidden_channels: List[int] = [64, 64, 128, 128, 256, 256],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        layers = []
        channels = [input_channels] + hidden_channels

        for i in range(len(hidden_channels)):
            dilation = 2 ** i
            layers.append(TemporalBlock(
                channels[i], channels[i + 1],
                kernel_size, dilation, dropout
            ))

        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        pooled = self.global_pool(features)
        pooled = pooled.squeeze(-1)
        logits = self.classifier(pooled)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_receptive_field(self) -> int:
        num_layers = len(self.tcn)
        kernel_size = 3
        return sum(2 ** i * (kernel_size - 1) for i in range(num_layers)) + 1


class SpeechIntentDecoder(nn.Module):
    """High-level decoder for speech intent classification."""

    VOCAB = ['yes', 'no', 'help', 'water', 'bathroom',
             'pain', 'tired', 'hungry', 'cold', 'hot']

    def __init__(
        self,
        input_channels: int = 32,
        vocab: Optional[List[str]] = None,
        window_size: int = 1000,
        device: str = 'cuda'
    ):
        super().__init__()

        self.vocab = vocab or self.VOCAB
        self.window_size = window_size
        self.device = device

        self.model = TemporalConvNet(
            input_channels=input_channels,
            num_classes=len(self.vocab)
        )
        self.model = self.model.to(device)

    def load_weights(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def decode(self, eeg_window: torch.Tensor) -> tuple:
        self.model.eval()

        if eeg_window.dim() == 2:
            eeg_window = eeg_window.unsqueeze(0)

        eeg_window = eeg_window.to(self.device)
        probs = self.model.predict(eeg_window)[0]

        confidence, idx = probs.max(dim=0)
        intent = self.vocab[idx.item()]
        prob_dict = {word: probs[i].item() for i, word in enumerate(self.vocab)}

        return intent, confidence.item(), prob_dict

    def export_onnx(self, path: str, opset_version: int = 14):
        self.model.eval()

        dummy_input = torch.randn(1, self.model.input_channels, self.window_size)
        dummy_input = dummy_input.to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=['eeg'],
            output_names=['logits'],
            dynamic_axes={
                'eeg': {0: 'batch', 2: 'time'},
                'logits': {0: 'batch'}
            },
            opset_version=opset_version
        )
        print(f"Exported ONNX model to {path}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TemporalConvNet(
        input_channels=32,
        num_classes=10
    ).to(device)

    x = torch.randn(1, 32, 1000).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=-1)

    print("TCN Model Test")
    print(f"  Device: {device}")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Receptive field: {model.get_receptive_field()} samples")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
