#!/usr/bin/env bash
# Download all models into models/ and the sherpa-onnx dynamic libraries into vendor/
# Requires: curl >= 7.71 (needs --retry-all-errors)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/whisper models/vad models/mt/en-zh models/mt/zh-en models/tts/zipvoice models/asr/sensevoice vendor/sherpa-onnx/lib

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Direct access to huggingface.co is often reset: defaults to the hf-mirror.com mirror;
# if the mirror/GitHub are both flaky, set PROXY=http://127.0.0.1:7897 (local proxy) and rerun
HF_BASE=${HF_BASE:-https://hf-mirror.com}
PROXY=${PROXY:-}
net_curl() {
  if [ -n "$PROXY" ]; then
    curl -x "$PROXY" "$@"
  else
    curl "$@"
  fi
}

# Download to .part first, then atomically rename on success, so a partial download is not
# mistaken for complete by the [ -s ] guard
dl() {
  [ -s "$2" ] && { echo "already have $2, skipping"; return; }
  echo "downloading $1"
  net_curl -L --fail --retry 3 --retry-all-errors -o "$2.part" "$1"
  mv -f "$2.part" "$2"
}

# 1) whisper small (f16, Metal)
dl "$HF_BASE/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" models/whisper/ggml-small.bin

# 2) silero-vad v5
dl https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx models/vad/silero_vad.onnx

# 3) opus-mt (Xenova ONNX export, int8 quantized)
for pair in en-zh zh-en; do
  base="$HF_BASE/Xenova/opus-mt-$pair/resolve/main"
  dl "$base/tokenizer.json" "models/mt/$pair/tokenizer.json"
  dl "$base/onnx/encoder_model_quantized.onnx" "models/mt/$pair/encoder_model_quantized.onnx"
  dl "$base/onnx/decoder_model_quantized.onnx" "models/mt/$pair/decoder_model_quantized.onnx"
done

# 4) ZipVoice (sherpa-onnx packaged): probe the exact asset name, then download
# Prefer the int8 quantized build (109MB vs 634MB for f16-distill — a big difference on slow
# networks); fall back to the first asset if absent
# All probe pipelines use || true: under set -o pipefail, grep exits 1 on no match and must
# not kill the script before its guard
if [ ! -s models/tts/zipvoice/tokens.txt ]; then
  ZIP_ASSETS=$(net_curl -sL https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models \
    | grep -o 'https://[^"]*zipvoice[^"]*zh-en[^"]*\.tar\.bz2' | grep -v multi || true)
  [ -n "$ZIP_ASSETS" ] || { echo "zipvoice asset not found; check https://github.com/k2-fsa/sherpa-onnx/releases/tags/tts-models"; exit 1; }
  ZIP_URL=$(echo "$ZIP_ASSETS" | grep int8 | head -1 || true)
  [ -n "$ZIP_URL" ] || ZIP_URL=$(echo "$ZIP_ASSETS" | head -1)
  echo "zipvoice asset: $ZIP_URL"
  net_curl -L --fail --retry 3 --retry-all-errors -o "$WORK/zipvoice.tar.bz2" "$ZIP_URL"
  mkdir -p "$WORK/zipvoice-x"
  tar -xjf "$WORK/zipvoice.tar.bz2" -C "$WORK/zipvoice-x"
  rm -rf models/tts/zipvoice && mkdir -p models/tts/zipvoice
  cp -R "$WORK"/zipvoice-x/*/* models/tts/zipvoice/
fi
# vocoder: the official archive does not include vocos; download it separately (referenced by Task 8 via --zipvoice-vocoder)
dl https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx \
  models/tts/zipvoice/vocos_24khz.onnx
echo "== zipvoice directory =="; ls -la models/tts/zipvoice/

# 5) SenseVoice-small ASR (int8, zh/en etc., ~155MB): probe the asset name, then download
# Pinned to the 2024-07-17 build: matches the vendored c-api.h (v1.13.8) doc examples for the
# most stable compatibility
# Large GitHub files often stall mid-stream: the .part file lives in the target dir (resumable
# across runs), and -C - + --retry makes each retry resume from the offset instead of restarting
if [ ! -s models/asr/sensevoice/model.int8.onnx ]; then
  SV_ASSETS=$(net_curl -sL https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/asr-models \
    | grep -o 'https://[^"]*sense-voice-zh-en-ja-ko-yue-int8-2024-07-17\.tar\.bz2' || true)
  [ -n "$SV_ASSETS" ] || { echo "sensevoice asset not found; check https://github.com/k2-fsa/sherpa-onnx/releases/tags/asr-models"; exit 1; }
  SV_URL=$(echo "$SV_ASSETS" | head -1)
  SV_PART=models/asr/sensevoice.download.part
  echo "sensevoice asset: $SV_URL"
  net_curl -L --fail --retry 8 --retry-all-errors --retry-delay 2 -C - -o "$SV_PART" "$SV_URL"
  mkdir -p "$WORK/sv-x"
  tar -xjf "$SV_PART" -C "$WORK/sv-x"
  rm -f "$SV_PART"
  rm -rf models/asr/sensevoice && mkdir -p models/asr/sensevoice
  cp -R "$WORK"/sv-x/*/* models/asr/sensevoice/
  echo "== sensevoice directory =="; ls -la models/asr/sensevoice/
fi

# 6) sherpa-onnx shared libraries (macOS universal2)
# v1.13.x asset name is osx-universal2-shared-lib.tar.bz2 (older releases: shared-libs)
if [ ! -s vendor/sherpa-onnx/lib/libsherpa-onnx-c-api.dylib ]; then
  LIB_URL=$(net_curl -sL https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/latest \
    | grep -o 'https://[^"]*osx-universal2-shared-lib[^"]*\.tar\.bz2' | head -1 || true)
  [ -n "$LIB_URL" ] || { echo "shared-libs asset not found; check the releases page"; exit 1; }
  echo "sherpa libs asset: $LIB_URL"
  net_curl -L --fail --retry 3 --retry-all-errors -o "$WORK/sherpa-libs.tar.bz2" "$LIB_URL"
  mkdir -p "$WORK/sherpa-libs-x"
  tar -xjf "$WORK/sherpa-libs.tar.bz2" -C "$WORK/sherpa-libs-x"
  cp "$WORK"/sherpa-libs-x/*/lib/*.dylib vendor/sherpa-onnx/lib/
  echo "re-signing dylibs (ad-hoc, required on Apple Silicon)..."
  for f in vendor/sherpa-onnx/lib/*.dylib; do codesign -f -s - "$f"; done
  echo "re-signing done"
fi
echo "== sherpa libs =="; ls vendor/sherpa-onnx/lib/

# 5) Qwen2.5-1.5B-Instruct (plan 7 text polish layer, GGUF Q4_K_M ~1GB; optional, not needed with polish off)
# Note: 0.5B (Q4/Q8) proved incapable of homophone repair in testing; bumped to 1.5B (selection notes in the polish-qwen crate docs)
mkdir -p models/polish/qwen1.5b
dl "$HF_BASE/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
   models/polish/qwen1.5b/model.gguf
dl "$HF_BASE/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json" \
   models/polish/qwen1.5b/tokenizer.json

echo "all set"
