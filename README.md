# Parrots

Real-time, on-device speech-to-speech translation for macOS. Speak in one
language; your listeners hear a cloned voice speaking another — in a meeting
app, over a virtual microphone. Their speech comes back to you translated and
spoken in your own voice.

Everything runs locally: speech recognition, machine translation, speech
synthesis, and (optionally) an LLM text-correction pass. No cloud services.

## Features

- **Two translation directions**
  - **A (speak → listeners)**: your microphone → ASR → MT → speech synthesized
    with *your* enrolled voice → played to a virtual microphone that the
    meeting app uses.
  - **B (listeners → you)**: the meeting app plays remote audio into a virtual
    speaker → Parrots captures it → ASR → MT → playback in your ears.
- **Incremental clause-by-clause pipeline**: translations start playing while
  you are still speaking (rolling ASR + VAD-based clause submission).
  - Submission triggers: sentence punctuation observed by the rolling ASR, or
    a ≥400 ms micro-pause after ≥6 content characters.
  - Anti-fragmentation: minimum 1 s of audio per clause, ≥2 content chars at
    the cut point, trailing punctuation never counts as a cut.
- **On-device models**
  - ASR: SenseVoice-small (default, int8 ONNX via sherpa-onnx) or
    whisper-small (Metal, whisper.cpp)
  - MT: opus-mt en↔zh (Xenova ONNX int8 exports)
  - TTS: ZipVoice zero-shot voice cloning (sherpa-onnx, int8)
  - Text polisher (optional): Qwen2.5-1.5B-Instruct GGUF via
    [candle](https://github.com/huggingface/candle) — fixes homophone typos
    and removes filler words before translation
- **Virtual audio devices**: branded CoreAudio HAL driver
  (`Parrots Microphone` 2ch, `Parrots Speakers` 16ch) with a one-click
  installer pkg
- **Device hot-swap**: live streams survive headphone unplug/reconnect and
  follow the default device (Bluetooth on/off) automatically
- **Echo cancellation**: system VoiceProcessingIO (FaceTime-grade AEC) capture
  path for `talk`, with automatic fallback
- **Hotword correction**: pinyin-based fuzzy matching fixes mis-recognized
  proper nouns (e.g. `稀有記` → `西游记`) before translation
- **Health check**: `parrots doctor` verifies devices, models, voice profiles
  and hotwords in one shot
- **Latency gates in CI**: first-audio ≤2.5s on fixture benchmarks; every
  stage timed and reported

## Requirements

- macOS 11+ (Apple Silicon tested; driver binaries are universal)
- Rust toolchain (edition 2021)
- Xcode (only for building the audio driver)
- ~2 GB disk for models (optional: +1 GB for the text polisher)

## Quick start

```bash
# 1. Build
cargo build --release

# 2. Download models (defaults to the hf-mirror.com mirror; see below)
./scripts/download-models.sh

# 3. Build & install the virtual audio driver (needs Xcode + admin password)
cd driver/macos
./make-pkg.sh                 # produces build/ParrotsAudio-<version>.pkg
open build/ParrotsAudio-0.1.0.pkg   # double-click install; restarts CoreAudio
cd ../..

# 4. Health check — everything should be OK (hotword/voice profile are WARN-only)
./target/release/parrots doctor
```

If Gatekeeper complains about the ad-hoc signed pkg, right-click → Open.

### Model downloads behind a firewall

`download-models.sh` uses `https://hf-mirror.com` by default. Alternatives:

```bash
HF_BASE=https://huggingface.co ./scripts/download-models.sh   # direct
PROXY=http://127.0.0.1:7897 ./scripts/download-models.sh      # local proxy
```

## Meeting-app configuration

In your meeting app (Zoom / Teams / 腾讯会议 / …):

| Setting | Choose | Effect |
|---|---|---|
| Microphone | `Parrots Microphone` | the meeting hears Parrots' translated voice |
| Speaker | `Parrots Speakers` | remote audio is fed to Parrots for translation |

To also hear remote participants locally, create a **Multi-Output Device**
in Audio MIDI Setup (headphones + `Parrots Speakers`) and point the meeting
app's speaker at it.

## Usage

```bash
# One-shot file translation (writes translated wav + latency report)
parrots translate --input meeting.wav --from en --to zh \
    --out translated.wav --report report.json

# Real time: remote speech (captured from the default/virtual input) → translation
parrots live --from en --to zh

# Real time: your speech → cloned voice in another language
parrots talk --from zh --to en

# Meeting combo: talk output goes to the virtual microphone
parrots talk --live --device "Parrots Microphone"
parrots live --device "Parrots Speakers"
```

### Voice enrollment (direction A)

```bash
parrots enroll --name my        # read the on-screen sentence; creates profiles/my.{wav,txt}
parrots talk --voice my
```

### Hotwords

```bash
parrots hotword 西游记          # append to profiles/hotwords.txt
parrots hotword 红楼梦
```

Hotwords fix mis-recognized proper nouns before translation (exact + pinyin
fuzzy matching, longest match first). The same list is also injected as
context into the text polisher and the whisper initial prompt.

### Text polisher

An optional LLM pass (Qwen2.5-1.5B, local candle inference) rewrites the
transcript before translation: homophone typos (with hotwords as context),
filler words (嗯/呃/然后), and sentence smoothing.

```bash
parrots translate --input x.wav --polish on     # force on (auto = on for files, off for live)
parrots live --polish on                        # live: bounded to 800 ms, auto = off
parrots translate --input x.wav --polish off
```

- Timeout is enforced token-by-token; on timeout the partial output (or the
  original text) is used, never blocking the pipeline.
- Model missing → the layer degrades to a no-op with a warning.
- Polish latency is reported as `polish_ms_mean` in `--report` JSON.

### Useful flags

| Flag | Applies to | Meaning |
|---|---|---|
| `--device <name>` | live/talk | strict input/output device; without it, live prefers `Parrots Speakers` and falls back to the system default |
| `--asr sensevoice\|whisper` | all | ASR engine (default sensevoice) |
| `--incremental <bool>` | live/talk | clause-by-clause pipeline (default true) |
| `--aec <bool>` | talk | VoiceProcessingIO echo-cancelling capture (default true, auto-fallback) |
| `--gate-playback <bool>` | talk | playback gate; auto = off while AEC is active |
| `--polish auto\|on\|off` | translate/live/talk | text polisher (auto = on for files, off for live) |
| `--max-seconds <n>` | live | auto-exit after n seconds (e2e testing) |
| `--out/--report` | translate/live/talk | write translated wav / latency JSON |

### ASR engine comparison

| Engine | Notes | Best for |
|---|---|---|
| `sensevoice` (default) | sherpa-onnx int8, zh/en/ja/ko/yue, built-in punctuation; zh CER 0.026 vs whisper 0.385, 3–6× faster | Chinese & mixed zh-en, incremental pipeline |
| `whisper` | whisper-small on Metal | fallback (`--asr whisper`) |

A/B benchmark: `cargo test -p parrots-cli --release --test asr_ab -- --ignored --nocapture`

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PARROTS_MODELS` | `models` | model root |
| `PARROTS_PROFILES` | `profiles` | voice profiles + hotwords.txt |
| `PARROTS_TTS_NUM_STEPS` | `4` | ZipVoice flow-matching steps (2–5; fewer = faster) |
| `HF_BASE` / `PROXY` | mirror / none | model download endpoint / proxy |

## Latency

Fixture benchmark gates (see `apps/translator-cli/tests/e2e.rs`):

- First synthesized audio (translation starts playing) ≤ 2.5 s
- Incremental mode first clause ≤ 4 s (TTS-dominated; design target 1.5 s)
- Text polisher adds ≤ 1.5 s on file mode / bounded to 0.8 s on live

## Development

```bash
cargo test --workspace                                        # unit tests (no models needed)
cargo test -p parrots-cli -- --ignored                        # fixture latency gates
cargo test --release -p parrots-cli -- --ignored incremental  # incremental benchmark
cargo test --release -p parrots-platform-macos -- --ignored loopback   # driver loopback (needs driver)
cargo test --release -p parrots-cli --test polish_ab -- --ignored --nocapture  # polisher A/B
```

`scripts/make-fixture.sh` generates the speech fixtures used by the benchmarks
(zh fixtures are synthesized with macOS `say`; not committed).

### Workspace layout

| Crate | Purpose |
|---|---|
| `crates/core` | traits (`AsrEngine`, `Translator`, `Synthesizer`, `TextPolisher`, `AudioPlatform`), audio/VAD primitives |
| `crates/vad` | Silero VAD wrapper, speech endpointing |
| `crates/asr-sensevoice` / `asr-whisper` | ASR engines |
| `crates/mt-opus` | opus-mt translation (session cache, clause translation) |
| `crates/tts-zipvoice` | ZipVoice cloning TTS (bounded prompt cache) |
| `crates/polish-qwen` | local LLM text polisher |
| `crates/pipeline` | direction A/B pipelines, incremental segmenter, hotwords |
| `crates/platform-macos` | cpal capture/playback, resampling, device watcher, VPIO AEC |
| `crates/engine` | language pack registry |
| `apps/translator-cli` | the `parrots` CLI |
| `driver/macos` | branded HAL audio driver (see licensing below) |

## License

- Engine, CLI and all Rust crates: **MIT** (see [LICENSE](LICENSE))
- `driver/macos`: **GPL-3.0** — a branded fork of
  [BlackHole](https://github.com/ExistentialAudio/BlackHole)
  (© Existential Audio Inc.). Upstream source snapshot and attribution are
  included in `driver/macos/blackhole-upstream/` and `driver/macos/NOTICE.md`,
  as required by the GPL. The engine communicates with the driver through
  standard CoreAudio APIs and is not a derivative work.

## Acknowledgments

- [BlackHole](https://existential.audio) by Existential Audio — the loopback
  driver upstream
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — SenseVoice, ZipVoice
  and opus-mt runtimes
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — Whisper on Metal
- [candle](https://github.com/huggingface/candle) — pure-Rust LLM inference
- [silero-vad](https://github.com/snakers4/silero-vad) — voice activity
  detection
- [Qwen2.5](https://huggingface.co/Qwen) — text polisher model (Apache-2.0)
