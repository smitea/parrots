# Homebrew distribution (custom tap)

This directory holds the Homebrew packaging for Parrots: a **formula**
(CLI + menubar app binaries) and a **cask** (the branded virtual audio
driver pkg). Homebrew core requires community traction before accepting
formulae, so Parrots is distributed through a personal tap.

## End-user experience (once published)

```bash
# one-time: add the tap (points at your homebrew-parrots repo)
brew tap YOUR_GH_USER/parrots https://github.com/YOUR_GH_USER/homebrew-parrots

# CLI + menubar app (builds from source, ~5-15 min on first install)
brew install YOUR_GH_USER/parrots/parrots

# virtual audio driver (signed pkg; macOS asks for the admin password)
brew install --cask YOUR_GH_USER/parrots/parrots-audio

# update / uninstall
brew upgrade
brew uninstall YOUR_GH_USER/parrots/parrots
brew uninstall --cask YOUR_GH_USER/parrots/parrots-audio
```

Models are not handled by Homebrew (~2 GB of local-inference weights);
users run `scripts/download-models.sh` from a clone of this repository
(see the formula caveats, printed on install).

## Publishing checklist

1. **Publish this repository** as `github.com/YOUR_GH_USER/parrots`
   (replace `YOUR_GH_USER` everywhere in this directory).
2. **Tag a release**: `git tag v0.1.0 && git push origin v0.1.0` — the
   `release` workflow builds the binaries and the driver pkg and publishes
   a GitHub Release containing:
   - `parrots-<version>.tar.gz` (source + vendored sherpa dylibs)
   - `ParrotsAudio-<version>.pkg`
   The workflow log prints the two SHA-256 sums.
3. **Create the tap repository**: `github.com/YOUR_GH_USER/homebrew-parrots`
   with layout:
   ```
   Formula/parrots.rb        <- copy from packaging/Formula/
   Casks/parrots-audio.rb    <- copy from packaging/Casks/
   ```
4. **Fill in the checksums** from the release workflow log into both files
   (and replace `YOUR_GH_USER`).
5. Verify from a clean machine:
   ```bash
   brew tap YOUR_GH_USER/parrots https://github.com/YOUR_GH_USER/homebrew-parrots
   brew install --build-from-source YOUR_GH_USER/parrots/parrots
   brew install --cask YOUR_GH_USER/parrots/parrots-audio
   parrots doctor
   ```

## Notes

- Source builds take a while (whisper-rs + egui + sherpa are large C/C++
  builds). Prebuilt bottles can be added later with `brew test-bot` in CI
  once the tap is public.
- The formula is arm64-only (`depends_on arch: :arm64`) — matching the
  driver and Metal-first model setup.
- Driver licensing: the pkg is a GPL-3.0 BlackHole derivative; its source
  ships inside this repository (`driver/macos/`), satisfying the GPL when
  distributing the pkg.
- `brew upgrade` picks up new versions automatically as long as each
  release bumps `version` in both files and the checksums are refreshed.
