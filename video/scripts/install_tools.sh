#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libpango1.0-dev libcairo2-dev pkg-config texlive-latex-base texlive-latex-extra dvisvgm fonts-inter fonts-jetbrains-mono ttyd
if ! command -v chromium >/dev/null 2>&1; then sudo snap install chromium; fi
../.venv/bin/python -m pip install -r requirements.txt
mkdir -p tools
arch="$(uname -m)"
case "$arch" in aarch64|arm64) suffix=arm64;; x86_64|amd64) suffix=x86_64;; *) echo "Unsupported VHS architecture: $arch" >&2; exit 1;; esac
archive="tools/vhs_0.12.1_Linux_${suffix}.tar.gz"
curl -fsSL "https://github.com/charmbracelet/vhs/releases/download/v0.12.1/vhs_0.12.1_Linux_${suffix}.tar.gz" -o "$archive"
tar -xzf "$archive" -C tools --strip-components=1 "vhs_0.12.1_Linux_${suffix}/vhs"
chmod +x tools/vhs
