#!/usr/bin/env bash
# Stat Lab voiceover: one-time setup.
# Installs the Python packages, makes sure espeak-ng is available, and downloads the
# Kokoro voice model (about 120 MB) into voiceover/models/. Safe to run again.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="$HERE/models"
REL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

echo "· Python packages"
pip install --quiet onnxruntime numpy scipy 2>/dev/null || pip install --quiet --break-system-packages onnxruntime numpy scipy

echo "· espeak-ng (turns text into pronunciation)"
if command -v espeak-ng >/dev/null 2>&1; then
  echo "  already installed: $(command -v espeak-ng)"
elif command -v apt-get >/dev/null 2>&1 && { [ "$(id -u)" = 0 ] || command -v sudo >/dev/null 2>&1; }; then
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq espeak-ng >/dev/null
elif command -v brew >/dev/null 2>&1; then
  brew install espeak-ng
else
  echo "  no package manager found, building from source into $HERE/espeak"
  git clone --depth 1 https://github.com/espeak-ng/espeak-ng.git "$HERE/espeak-src"
  cmake -S "$HERE/espeak-src" -B "$HERE/espeak-src/build" -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_ASYNC=OFF \
    -DCMAKE_INSTALL_PREFIX="$HERE/espeak" >/dev/null
  cmake --build "$HERE/espeak-src/build" -j4 >/dev/null
  cmake --install "$HERE/espeak-src/build" >/dev/null
  echo "  built: $HERE/espeak/bin/espeak-ng (make_voiceover.py finds it automatically)"
fi

echo "· Kokoro voice model"
mkdir -p "$MODELS"
for f in kokoro-v1.0.int8.onnx voices-v1.0.bin; do
  if [ -s "$MODELS/$f" ]; then echo "  $f already downloaded"; else
    curl -fL --retry 3 -o "$MODELS/$f" "$REL/$f"; fi
done
echo "Done. Try:  python voiceover/make_voiceover.py voiceover/scripts/max_interlagos_2m15.txt"
