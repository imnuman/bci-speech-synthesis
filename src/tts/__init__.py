"""Text-to-Speech Module - Natural voice synthesis for BCI output"""

from .kokoro import KokoroTTS
from .alsa_output import ALSAOutput

__all__ = ['KokoroTTS', 'ALSAOutput']
