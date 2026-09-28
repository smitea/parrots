#!/usr/bin/env bash
# Install the official BlackHole 2ch/16ch virtual audio driver (GPL-3.0; installed by the user, not distributed by Parrots)
#
# Usage:
#   ./scripts/install-blackhole.sh           # detect missing casks and install via brew (may prompt for password)
#   ./scripts/install-blackhole.sh --check   # detection only, no install (for automation/health checks)
#
# Structure: detect -> brew first / pkg download fallback -> post-install verification.
# This script embeds no passwords and never silently escalates privileges; password prompts
# come from the official brew/pkg installers.
set -euo pipefail
cd "$(dirname "$0")/.."

CHECK=0
if [ "${1:-}" = "--check" ]; then
  CHECK=1
fi

have() { command -v "$1" >/dev/null 2>&1; }

# Detection basis: loaded CoreAudio audio device names (matches the official installer: BlackHole 2ch / BlackHole 16ch)
installed() { system_profiler SPAudioDataType 2>/dev/null | grep -q "$1"; }

missing=()
installed "BlackHole 2ch"  || missing+=("blackhole-2ch")
installed "BlackHole 16ch" || missing+=("blackhole-16ch")

if [ "${#missing[@]}" -eq 0 ]; then
  echo "BlackHole 2ch/16ch are both installed"
  exit 0
fi

echo "to install: ${missing[*]}"
if [ "$CHECK" -eq 1 ]; then
  echo "(--check mode: detection only, nothing installed)"
  exit 1
fi

if have brew; then
  echo "installing via Homebrew (any password/authorization prompt is from the official installer)..."
  for c in "${missing[@]}"; do
    brew install --cask "$c"
  done
else
  # No brew: download the official pkg to ~/Downloads for the user to double-click (installer asks for the admin password)
  echo "Homebrew not found; downloading the official pkg instead (double-click to install after download):"
  for c in "${missing[@]}"; do
    case "$c" in
      blackhole-2ch)  url="https://existential.audio/downloads/BlackHole2ch-0.7.1.pkg" ;;
      blackhole-16ch) url="https://existential.audio/downloads/BlackHole16ch-0.7.1.pkg" ;;
      *) url="" ;;
    esac
    if [ -n "$url" ]; then
      dest="$HOME/Downloads/$(basename "$url")"
      echo "downloading $url -> $dest"
      curl -L --fail --retry 3 -o "$dest.part" "$url"
      mv -f "$dest.part" "$dest"
    else
      echo "unknown component $c; download it manually from https://existential.audio/blackhole/"
    fi
  done
  echo "Download complete. Double-click the pkg in ~/Downloads to install (or get the latest from https://existential.audio/blackhole/),"
  echo "then rerun: ./scripts/install-blackhole.sh --check"
  exit 1
fi

# Post-install verification: the brew/pkg installers usually restart coreaudiod automatically; in rare cases log out and back in
if installed "BlackHole 2ch" && installed "BlackHole 16ch"; then
  echo "install complete: BlackHole 2ch and BlackHole 16ch are both ready"
else
  echo "no devices detected after install; try in order:"
  echo "  1. sudo killall coreaudiod   # restart the CoreAudio daemon, then re-check"
  echo "  2. ./scripts/install-blackhole.sh --check"
  echo "  3. log out and back in (or reboot), then check again (brew also notes a restart may be needed)"
  exit 1
fi
