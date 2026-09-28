#!/usr/bin/env bash
# Generate direction-B test audio with the macOS built-in say (16kHz mono wav)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p fixtures
[ -s fixtures/en-hello.wav ] || {
  say -o /tmp/en-hello.aiff "Hello, how are you doing today? I hope everything is going well."
  afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/en-hello.aiff fixtures/en-hello.wav
}
EN_HELLO_TEXT="Hello, how are you doing today? I hope everything is going well."
[ -s fixtures/en-hello.txt ] || printf '%s' "$EN_HELLO_TEXT" > fixtures/en-hello.txt
[ -s fixtures/en-meeting.wav ] || {
  say -o /tmp/en-meeting.aiff "Hi everyone, thanks for joining. Today we will discuss the quarterly roadmap and the new feature launch timeline."
  afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/en-meeting.aiff fixtures/en-meeting.wav
}
EN_MEETING_TEXT="Hi everyone, thanks for joining. Today we will discuss the quarterly roadmap and the new feature launch timeline."
[ -s fixtures/en-meeting.txt ] || printf '%s' "$EN_MEETING_TEXT" > fixtures/en-meeting.txt
# Chinese fixture: prefer Tingting/Meijia (standard zh_CN voices); extract the full voice name
# (may contain spaces; trailing whitespace stripped) to avoid picking an English voice of the same name
ZH_VOICE=$(say -v '?' | grep zh_CN | grep -E '^(Tingting|Meijia) ' | head -1 | sed -E 's/[[:space:]]*zh_CN.*//' || true)
[ -n "$ZH_VOICE" ] || ZH_VOICE=$(say -v '?' | grep zh_CN | head -1 | sed -E 's/[[:space:]]*zh_CN.*//' || true)
[ -n "$ZH_VOICE" ] || { echo "no Chinese TTS voice found (say -v ? has no zh_CN)"; exit 1; }
[ -s fixtures/zh-hello.wav ] || {
  say -v "$ZH_VOICE" -o /tmp/zh-hello.aiff "你好,很高兴认识大家。今天过得怎么样?"
  afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/zh-hello.aiff fixtures/zh-hello.wav
  printf '你好,很高兴认识大家。今天过得怎么样?' > fixtures/zh-hello.txt
}
[ -s fixtures/zh-meeting.wav ] || {
  say -v "$ZH_VOICE" -o /tmp/zh-meeting.aiff "大家好,谢谢大家参加今天的会议。我们将讨论本季度的产品路线图,以及新功能的上线时间表。"
  afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/zh-meeting.aiff fixtures/zh-meeting.wav
  printf '大家好,谢谢大家参加今天的会议。我们将讨论本季度的产品路线图,以及新功能的上线时间表。' > fixtures/zh-meeting.txt
}
# Long Chinese passage (~20s of continuous speech; baseline for incremental-mode latency)
ZH_LONG_TEXT="大家好,今天很高兴和大家分享我们这个季度的进展。首先我们来看整体的业务数据,用户的增长速度比上个季度提升了不少。接下来我会介绍新版本的几个重点功能,包括实时翻译、离线模式和多人协作。最后我们再讨论一下接下来的产品路线图,以及团队下一步的工作安排。"
[ -s fixtures/zh-long.wav ] || {
  say -v "$ZH_VOICE" -o /tmp/zh-long.aiff "$ZH_LONG_TEXT"
  afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/zh-long.aiff fixtures/zh-long.wav
  printf '%s' "$ZH_LONG_TEXT" > fixtures/zh-long.txt
}
ls -la fixtures/
