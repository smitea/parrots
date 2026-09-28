class Parrots < Formula
  desc "Real-time on-device speech-to-speech translation for macOS"
  homepage "https://github.com/smitea/parrots"
  url "https://github.com/smitea/parrots/releases/download/v#{version}/parrots-#{version}.tar.gz"
  version "0.1.0"
  sha256 "replace_with_release_sha256" # printed by the release workflow
  license "MIT"

  # Apple Silicon only: the audio driver, Metal ASR/TTS and the VPIO AEC
  # path are all arm64-first (the driver binaries are universal, but the
  # packaging targets Apple Silicon Macs).
  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on macos: :monterey

  def install
    # sherpa-onnx dylibs are vendored in the release tarball; link them into
    # prefix/lib and point the binaries at it with an extra rpath.
    lib.install Dir["vendor/sherpa-onnx/lib/*.dylib"]

    system "cargo", "install", *std_cargo_args(path: "apps/translator-cli")
    system "cargo", "install", *std_cargo_args(path: "app")

    # cargo links with @rpath; brew's prefix is outside the build tree, so
    # append our lib dir to each binary and re-sign (ad-hoc) afterwards.
    require "macho"
    [bin/"parrots", bin/"parrots-app"].each do |b|
      MachO::Tools.add_rpath(b, lib)
      system "codesign", "-f", "-s", "-", b
    end
  end

  def caveats
    <<~EOS
      Speech models are NOT installed by this formula (~2 GB, local inference).
      Download them once with:

        git clone https://github.com/smitea/parrots.git
        cd parrots && ./scripts/download-models.sh

      Meeting apps must use "Parrots Microphone" as their microphone and
      "Parrots Speakers" as their speaker; install the virtual audio driver
      with:  brew install --cask smitea/parrots/parrots-audio
      Run `parrots doctor` to verify the setup.
    EOS
  end

  test do
    assert_match "Usage:", shell_output("#{bin}/parrots --help")
    assert_match "Parrots", shell_output("#{bin}/parrots-app --help 2>&1; true")
  end
end
