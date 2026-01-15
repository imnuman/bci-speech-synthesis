"""Text-to-Speech Module - Natural voice synthesis for BCI output"""

from .kokoro import KokoroTTS, get_response_text
from .alsa_output import ALSAOutput

__all__ = ['KokoroTTS', 'ALSAOutput', 'get_response_text']
