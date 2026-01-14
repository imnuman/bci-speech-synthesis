"""
SPI Reader for JNEEG 32-Channel EEG Shield
Interfaces with 4x ADS1299 ADCs via Linux spidev
"""

import numpy as np
import threading
import time
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class ADS1299Config:
    """ADS1299 ADC configuration"""
    sample_rate: int = 1000  # Hz (250, 500, 1000, 2000, 4000, 8000, 16000)
    gain: int = 24           # PGA gain (1, 2, 4, 6, 8, 12, 24)
    channels: int = 8        # Channels per chip
    resolution: int = 24     # Bit depth


class SPIReader:
    """
    High-speed SPI reader for JNEEG 32-channel EEG shield.

    The JNEEG shield uses 4x TI ADS1299 ADCs daisy-chained on SPI.
    Each ADS1299 provides 8 channels of 24-bit data.

    Hardware connection:
        - SPI0 CE0: /dev/spidev0.0
        - DRDY pin: GPIO interrupt for data ready

    Data format:
        - 24-bit signed integer per channel
        - LSB = Vref / (2^23 - 1) / Gain
        - Default: 4.5V / 8388607 / 24 = 22.35 nV/count
    """

    # ADS1299 register addresses
    REG_CONFIG1 = 0x01
    REG_CONFIG2 = 0x02
    REG_CONFIG3 = 0x03
    REG_CH1SET = 0x05

    # Sample rate configuration (CONFIG1 register)
    SAMPLE_RATES = {
        16000: 0x00,
        8000: 0x01,
        4000: 0x02,
        2000: 0x03,
        1000: 0x04,
        500: 0x05,
        250: 0x06,
    }

    def __init__(
        self,
        device: str = '/dev/spidev0.0',
        sample_rate: int = 1000,
        channels: int = 32,
        speed_hz: int = 4000000,
        drdy_pin: int = 17
    ):
        """
        Initialize SPI reader.

        Args:
            device: SPI device path
            sample_rate: Sampling rate in Hz
            channels: Number of EEG channels (8, 16, 24, or 32)
            speed_hz: SPI clock speed
            drdy_pin: GPIO pin for data ready interrupt
        """
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.speed_hz = speed_hz
        self.drdy_pin = drdy_pin

        self.num_chips = channels // 8
        self.spi = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None

        # Conversion factor: 24-bit to microvolts
        # Vref = 4.5V, Gain = 24, Resolution = 2^23
        self.uv_per_count = (4.5 / (2**23 - 1) / 24) * 1e6

    def open(self) -> bool:
        """Open SPI device and configure ADCs."""
        try:
            import spidev
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)  # Bus 0, Device 0
            self.spi.max_speed_hz = self.speed_hz
            self.spi.mode = 0b01  # CPOL=0, CPHA=1

            self._configure_ads1299()
            return True

        except Exception as e:
            print(f"Failed to open SPI device: {e}")
            return False

    def _configure_ads1299(self):
        """Configure all ADS1299 chips."""
        if self.spi is None:
            return

        # Stop continuous mode
        self._send_command(0x11)  # SDATAC

        # Set sample rate
        rate_bits = self.SAMPLE_RATES.get(self.sample_rate, 0x04)
        self._write_register(self.REG_CONFIG1, 0x90 | rate_bits)

        # Enable internal reference
        self._write_register(self.REG_CONFIG3, 0xE0)

        # Configure all channels: Gain=24, Normal input
        for ch in range(8):
            self._write_register(self.REG_CH1SET + ch, 0x60)

        # Start continuous mode
        self._send_command(0x10)  # RDATAC

    def _send_command(self, cmd: int):
        """Send command to ADS1299."""
        if self.spi:
            self.spi.xfer2([cmd])
            time.sleep(0.001)

    def _write_register(self, reg: int, value: int):
        """Write to ADS1299 register."""
        if self.spi:
            # WREG command: 0x40 | register
            self.spi.xfer2([0x40 | reg, 0x00, value])
            time.sleep(0.001)

    def _read_sample(self) -> np.ndarray:
        """
        Read one sample from all channels.

        Returns:
            Array of shape (channels,) with values in microvolts
        """
        if self.spi is None:
            return np.zeros(self.channels)

        # Each chip: 3 bytes status + 3 bytes * 8 channels = 27 bytes
        bytes_per_chip = 27
        total_bytes = bytes_per_chip * self.num_chips

        # Read raw bytes
        raw = self.spi.xfer2([0x00] * total_bytes)

        # Parse 24-bit samples
        samples = np.zeros(self.channels)
        for chip in range(self.num_chips):
            offset = chip * bytes_per_chip + 3  # Skip status bytes
            for ch in range(8):
                idx = offset + ch * 3
                # 24-bit signed integer (big-endian)
                value = (raw[idx] << 16) | (raw[idx+1] << 8) | raw[idx+2]
                if value & 0x800000:  # Sign extend
                    value -= 0x1000000
                samples[chip * 8 + ch] = value * self.uv_per_count

        return samples

    def start(self, callback: Callable[[np.ndarray], None]):
        """
        Start continuous data acquisition.

        Args:
            callback: Function called with each sample array
        """
        if self._running:
            return

        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self._thread.start()

    def _acquisition_loop(self):
        """Main acquisition loop."""
        interval = 1.0 / self.sample_rate
        next_time = time.perf_counter()

        while self._running:
            sample = self._read_sample()

            if self._callback:
                self._callback(sample)

            # Maintain precise timing
            next_time += interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        """Stop data acquisition."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def close(self):
        """Close SPI device."""
        self.stop()
        if self.spi:
            self.spi.close()
            self.spi = None


if __name__ == '__main__':
    # Test with simulated data (no hardware)
    reader = SPIReader(sample_rate=250, channels=32)

    samples_received = []
    def on_sample(sample):
        samples_received.append(sample)

    print("SPIReader module loaded successfully")
    print(f"Configuration: {reader.channels} channels @ {reader.sample_rate} Hz")
    print(f"Resolution: {reader.uv_per_count:.4f} uV/count")
