#!/bin/zsh
# One-time setup for Grab. Run:  ./setup.sh
set -e
cd "$(dirname "$0")"

echo "==> Python venv + dependencies"
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip flask yt-dlp certifi

if ! command -v ffmpeg >/dev/null; then
  echo "==> Installing ffmpeg (via Homebrew)"
  brew install ffmpeg
fi

if [ ! -f models/ggml-base.bin ]; then
  echo "==> Downloading Whisper model (~142 MB, one-time)"
  mkdir -p models
  curl -L -o models/ggml-base.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
fi

if [ ! -x bin/whisper-cli ]; then
  echo "==> Building whisper.cpp natively (needs cmake + Xcode CLT)"
  command -v cmake >/dev/null || brew install cmake
  rm -rf /tmp/whispercpp
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp /tmp/whispercpp
  EXTRA_FLAGS=""
  [ "$(uname -m)" = "arm64" ] && EXTRA_FLAGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DGGML_NATIVE=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON"
  cmake -S /tmp/whispercpp -B /tmp/whispercpp/build \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release ${=EXTRA_FLAGS}
  cmake --build /tmp/whispercpp/build -j --config Release
  mkdir -p bin
  cp /tmp/whispercpp/build/bin/whisper-cli bin/
fi

echo ""
echo "✅ Done. Start the app with:  ./start.command"
