#!/usr/bin/env bash
# Build the Parrots-branded virtual audio drivers (two variants)
#   ParrotsMicrophone.driver  2ch  (direction A exit: meeting-app microphone)
#   ParrotsSpeakers.driver    16ch (direction B entry: meeting-app speaker)
# Derived from BlackHole v0.7.1 (GPL-3.0, see NOTICE.md); renaming is done via build-time
# constant overrides — upstream official customization guide: README "Renaming BlackHole".
set -euo pipefail
cd "$(dirname "$0")"

SRC="src"
OUT="build"
DD="$OUT/DerivedData"
mkdir -p "$OUT"

build_variant() {
  local product="$1" channels="$2" bundle="$3"
  local part2
  case "$product" in
    ParrotsMicrophone) part2="Microphone" ;;
    ParrotsSpeakers)   part2="Speakers" ;;
    *) part2="$product" ;;
  esac
  local label="Parrots $part2"
  echo "==> building $product.driver (${channels}ch, $label)"

  # Generate the variant header (string constants go through a C header to avoid xcodebuild CLI quoting issues)
  cat > "$SRC/Parrots/variant.h" <<EOF
// auto-generated (build.sh) — do not edit by hand
#define kDriver_Name            "$label"
#define kDevice_Name            "$label"
#define kDevice2_Name           "$label 2"
#define kHas_Driver_Name_Format false
#define kNumber_Of_Channels     $channels
#define kPlugIn_BundleID        "$bundle"
#define kPlugIn_Icon            "Parrots.icns"
#define kManufacturer_Name      "Parrots"
EOF

  xcodebuild \
    -project "$SRC/Parrots.xcodeproj" \
    -scheme Parrots \
    -configuration Release \
    -derivedDataPath "$DD" \
    PRODUCT_NAME="$product" \
    PRODUCT_BUNDLE_IDENTIFIER="$bundle" \
    CURRENT_PROJECT_VERSION=100 \
    MARKETING_VERSION=0.1.0 \
    MACOSX_DEPLOYMENT_TARGET=11.0 \
    DEVELOPMENT_TEAM="" \
    CODE_SIGN_IDENTITY="-" \
    CODE_SIGN_STYLE=Manual \
    build >/dev/null

  local built="$DD/Build/Products/Release/$product.driver"
  [ -d "$built" ] || { echo "missing build artifact: $built"; exit 1; }
  rm -rf "$OUT/$product.driver"
  cp -R "$built" "$OUT/$product.driver"

  # ad-hoc signing (local/internal testing; Developer ID comes at M4)
  codesign --force --sign - "$OUT/$product.driver"

  # Structural verification
  [ -f "$OUT/$product.driver/Contents/Info.plist" ] || { echo "Info.plist missing"; exit 1; }
  [ -x "$OUT/$product.driver/Contents/MacOS/$product" ] || { echo "executable missing"; exit 1; }
  codesign -dv "$OUT/$product.driver" 2>&1 | grep -q "Identifier=$bundle" \
    || { echo "signing Identifier does not match bundle id"; exit 1; }
  strings "$OUT/$product.driver/Contents/MacOS/$product" | grep -q "$label" \
    || { echo "device name $label not embedded in binary"; exit 1; }
  echo "    OK: $OUT/$product.driver"
}

build_variant ParrotsMicrophone 2  audio.parrots.microphone
build_variant ParrotsSpeakers   16 audio.parrots.speakers

# Residue check: build artifacts must not contain "BlackHole" (LICENSE excepted — GPL attribution requires keeping it)
if grep -r "BlackHole" --exclude=LICENSE "$OUT"/*.driver >/dev/null 2>&1; then
  echo "BlackHole residue found in build artifacts:"; grep -rl "BlackHole" "$OUT"/*.driver; exit 1
fi

echo "all variants built"
