# NOTICE

本目录(`driver/macos/`)包含 BlackHole 的派生修改版本。

## 上游归属

- 项目:BlackHole
- 版权:© 2019–2026 Existential Audio Inc.
- 许可证:GPL-3.0(全文见本目录 `LICENSE`)
- 上游版本:v0.7.1(`blackhole-upstream/VERSION`)
- 上游地址:https://github.com/ExistentialAudio/BlackHole

## 派生修改声明

`blackhole-upstream/` 为原封不动的上游快照(v0.7.1)。
针对 Parrots 的品牌化修改(设备名、bundle id、插件名等)以补丁/覆盖形式
应用于构建流程(见 `build.sh`),不直接改写上游快照文件,
以便对照上游与跟踪差异。

全部修改点清单见 `build.sh` 内的改名表与仓库提交历史。

## 许可证边界

- `driver/macos/` 内的驱动派生作品依 **GPL-3.0** 单独开源。
- Parrots 引擎/App(crate 与 app 层 Rust 代码)经标准 CoreAudio API
  与系统音频服务通信,不与驱动链接,不属于派生作品,维持各自许可。
