#!/usr/bin/env bash
# Uninstall the Parrots virtual audio drivers (requires sudo)
# Usage: sudo ./uninstall.sh
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: sudo $0"; exit 1; }

rm -rf "/Library/Audio/Plug-Ins/HAL/ParrotsMicrophone.driver" \
       "/Library/Audio/Plug-Ins/HAL/ParrotsSpeakers.driver"

killall coreaudiod 2>/dev/null || true
echo "Parrots virtual audio drivers uninstalled (Parrots Microphone / Parrots Speakers)"
