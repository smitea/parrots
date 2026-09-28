#!/usr/bin/env bash
# Package the ParrotsAudio one-shot installer pkg (both driver variants + postinstall that restarts CoreAudio)
# Usage: ./make-pkg.sh [version]   (default 0.1.0)
set -euo pipefail
cd "$(dirname "$0")"

VERSION="${1:-0.1.0}"
OUT="build"
STAGE="$OUT/pkg-root"
PKG="$OUT/ParrotsAudio-$VERSION.pkg"

# Depends on the two-variant build artifacts
if [ ! -d "$OUT/ParrotsMicrophone.driver" ] || [ ! -d "$OUT/ParrotsSpeakers.driver" ]; then
  echo "build artifacts missing; building first..."
  ./build.sh
fi

rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -R "$OUT/ParrotsMicrophone.driver" "$OUT/ParrotsSpeakers.driver" "$STAGE/"

pkgbuild \
  --root "$STAGE" \
  --scripts pkg \
  --identifier audio.parrots.driver \
  --version "$VERSION" \
  --install-location "/Library/Audio/Plug-Ins/HAL" \
  "$PKG"

echo "==> $PKG"
pkgutil --expand-full "$PKG" "$OUT/pkg-inspect" >/dev/null 2>&1 || true
[ -f "$PKG" ] || { echo "pkg packaging failed"; exit 1; }
rm -rf "$OUT/pkg-inspect"
ls -lh "$PKG"
