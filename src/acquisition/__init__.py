"""EEG Data Acquisition Module - SPI interface for JNEEG shield"""

from .spi_reader import SPIReader
from .ring_buffer import RingBuffer
from .lsl_outlet import LSLOutlet

__all__ = ['SPIReader', 'RingBuffer', 'LSLOutlet']
