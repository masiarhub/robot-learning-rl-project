#!/bin/bash
set -e
git config --global user.email "aron.stettler@gmail.com"
git config --global user.name "Aron"
cd "$(dirname "$0")/isaac_so_arm101-main"
sudo apt-get install -y libglu1-mesa
uv pip install "setuptools==59.8.0" wheel
.venv/bin/pip install flatdict==4.0.1 --no-build-isolation
uv sync
