#!/bin/zsh
# Double-click this file in Finder to start Grab.
cd "$(dirname "$0")"
.venv/bin/pip install --quiet --upgrade yt-dlp   # platforms change often; stay current
exec .venv/bin/python app.py
