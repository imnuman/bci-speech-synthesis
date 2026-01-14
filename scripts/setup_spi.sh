#!/bin/bash
# Setup SPI interface on Jetson Orin NX for JNEEG EEG Shield
# Run with sudo

set -e

echo "=== BCI Speech Synthesis - SPI Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: Not running on NVIDIA Jetson"
    echo "SPI configuration may differ on other platforms"
fi

echo "[1/5] Enabling SPI interface..."

# Enable SPI via device tree
if [ -f /opt/nvidia/jetson-io/jetson-io.py ]; then
    echo "  Using jetson-io utility..."
    # This requires interactive mode, show instructions instead
    echo ""
    echo "  To enable SPI, run:"
    echo "    sudo /opt/nvidia/jetson-io/jetson-io.py"
    echo "  Then select: Configure 40-pin expansion header"
    echo "  Enable: spi1 (for /dev/spidev0.0)"
    echo ""
else
    echo "  jetson-io.py not found, checking alternative methods..."
fi

echo "[2/5] Loading SPI kernel module..."
modprobe spidev || echo "  spidev already loaded or not available"

echo "[3/5] Checking SPI device..."
if [ -e /dev/spidev0.0 ]; then
    echo "  /dev/spidev0.0 found!"
    ls -la /dev/spidev0.0
else
    echo "  /dev/spidev0.0 not found"
    echo "  Available SPI devices:"
    ls -la /dev/spidev* 2>/dev/null || echo "    None"
fi

echo "[4/5] Setting up udev rules..."

# Create udev rule for SPI access without sudo
UDEV_RULE="/etc/udev/rules.d/99-spidev.rules"
cat > "$UDEV_RULE" << 'EOF'
# Allow user access to SPI devices
SUBSYSTEM=="spidev", MODE="0666"

# JNEEG EEG Shield - assign consistent name
KERNEL=="spidev0.0", SYMLINK+="eeg_shield"
EOF

echo "  Created $UDEV_RULE"

# Reload udev rules
udevadm control --reload-rules
udevadm trigger

echo "[5/5] Configuring GPIO for DRDY pin..."

# Export GPIO17 for data ready interrupt
GPIO_PIN=17
GPIO_PATH="/sys/class/gpio"

if [ ! -d "$GPIO_PATH/gpio$GPIO_PIN" ]; then
    echo $GPIO_PIN > "$GPIO_PATH/export" 2>/dev/null || true
fi

if [ -d "$GPIO_PATH/gpio$GPIO_PIN" ]; then
    echo "in" > "$GPIO_PATH/gpio$GPIO_PIN/direction"
    echo "falling" > "$GPIO_PATH/gpio$GPIO_PIN/edge"
    chmod 666 "$GPIO_PATH/gpio$GPIO_PIN/value"
    echo "  GPIO$GPIO_PIN configured for DRDY interrupt"
else
    echo "  Warning: Could not configure GPIO$GPIO_PIN"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "SPI Configuration:"
echo "  Device: /dev/spidev0.0"
echo "  Mode: 1 (CPOL=0, CPHA=1)"
echo "  Max Speed: 4 MHz"
echo ""
echo "GPIO Configuration:"
echo "  DRDY: GPIO$GPIO_PIN (Pin 11)"
echo ""
echo "To verify SPI communication:"
echo "  python -c \"import spidev; s=spidev.SpiDev(); s.open(0,0); print('SPI OK')\""
echo ""
