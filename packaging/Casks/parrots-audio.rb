cask "parrots-audio" do
  version "0.1.0"
  sha256 "REPLACE_WITH_RELEASE_SHA256" # printed by the release workflow

  url "https://github.com/YOUR_GH_USER/parrots/releases/download/v#{version}/ParrotsAudio-#{version}.pkg"
  name "Parrots Audio Driver"
  desc "Virtual audio devices for the Parrots translator"
  homepage "https://github.com/YOUR_GH_USER/parrots"

  livecheck do
    url :homepage
    strategy :github_latest
  end

  depends_on macos: :monterey

  pkg "ParrotsAudio-#{version}.pkg"

  # The pkg's postinstall restarts coreaudiod during install (admin prompt
  # is handled by the macOS installer); the pkgs install both HAL drivers.
  uninstall launchctl: "system|com.apple.audio.coreaudiod",
            script:    {
              executable: "/usr/bin/killall",
              args:       ["coreaudiod"],
            },
            pkgutil:   "audio.parrots.driver",
            delete:    [
              "/Library/Audio/Plug-Ins/HAL/ParrotsMicrophone.driver",
              "/Library/Audio/Plug-Ins/HAL/ParrotsSpeakers.driver",
            ]

  zap delete: [
    "/Library/Audio/Plug-Ins/HAL/ParrotsMicrophone.driver",
    "/Library/Audio/Plug-Ins/HAL/ParrotsSpeakers.driver",
  ]

  caveats <<~EOS
    Installs two CoreAudio HAL plug-ins into /Library/Audio/Plug-Ins/HAL:
      Parrots Microphone (2ch) — direction A exit (meeting app microphone)
      Parrots Speakers  (16ch) — direction B entry (meeting app speaker)

    The driver is a GPL-3.0 fork of BlackHole (source included in the
    parrots repository under driver/macos/).

    If the devices do not appear after install/uninstall, CoreAudio may need
    one more restart:  sudo killall coreaudiod
    Verify with:        parrots doctor
  EOS
end
