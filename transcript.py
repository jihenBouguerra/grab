"""Transcript pipeline for Grab — clean reimplementation.

One entry point: get_transcript(url, tmp, report) -> (text, source)

Strategy:
  1. Probe the link once for metadata (duration + which captions exist).
  2. If captions exist in a wanted language, download exactly that one file.
  3. Otherwise download the audio and transcribe locally with whisper.cpp.

Every stage logs what it's doing, has a hard timeout, and failures raise
TranscriptError with a message meant for the user's eyes.
"""

import glob
import logging
import os
import re
import subprocess
import threading

import yt_dlp

log = logging.getLogger("grab.transcript")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL = os.path.join(APP_DIR, "models", "ggml-base.bin")
_local = os.path.join(APP_DIR, "bin", "whisper-cli")
WHISPER_BIN = _local if os.path.exists(_local) else None
COOKIES = os.path.join(APP_DIR, "cookies.txt")  # optional, helps Instagram

CAPTION_LANGS = ["en", "de", "ar", "fr"]   # preference order
MAX_DURATION = 30 * 60                     # refuse longer whisper jobs


class TranscriptError(Exception):
    """User-readable failure."""


def _opts(tmp=None):
    o = {"quiet": True, "noprogress": True, "no_warnings": True,
         "noplaylist": True, "socket_timeout": 30}
    if tmp:
        o["outtmpl"] = os.path.join(tmp, "media.%(ext)s")
    if os.path.exists(COOKIES):
        o["cookiefile"] = COOKIES
    return o


def get_transcript(url, tmp, report):
    report("Reading link", None)
    info = _probe(url)
    duration = info.get("duration")
    title = info.get("title", "?")
    log.info("script: %s (%ss) %s", title[:60], duration, url)

    lang = _best_caption_lang(info)
    if lang:
        report("Fetching captions", None)
        text = _fetch_captions(url, tmp, lang)
        if text:
            log.info("script done via captions[%s]: %d chars", lang, len(text))
            return text, "captions"
        log.warning("captions[%s] advertised but unusable, falling back", lang)

    if not WHISPER_BIN or not os.path.exists(WHISPER_MODEL):
        raise TranscriptError("This video has no captions, and local "
                              "transcription isn't set up (run ./setup.sh).")
    if duration and duration > MAX_DURATION:
        raise TranscriptError(
            f"This video is {duration // 60} min long and has no captions — "
            f"transcription is capped at {MAX_DURATION // 60} min.")

    report("Downloading audio", 0)
    wav = _fetch_audio(url, tmp, report)
    report("Transcribing", 25)
    text = _whisper(wav, tmp, duration or 600, report)
    if not text:
        raise TranscriptError("Transcription produced no text — "
                              "the video may have no speech.")
    log.info("script done via whisper: %d chars", len(text))
    return text, "transcription"


# ---------------------------------------------------------------- stages ---
def _probe(url):
    try:
        with yt_dlp.YoutubeDL(_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        log.warning("probe failed: %s", e)
        raise TranscriptError(_friendly(e)) from e
    if info.get("_type") == "playlist":
        info = (info.get("entries") or [{}])[0]
    return info


def _best_caption_lang(info):
    """Pick the best available caption language from the probe — so we ask
    the platform for exactly one subtitle file (asking broadly = HTTP 429)."""
    for source in ("subtitles", "automatic_captions"):  # manual first
        available = info.get(source) or {}
        for want in CAPTION_LANGS:
            for code in available:
                if code == want or code.startswith(want + "-"):
                    return code
    return None


def _fetch_captions(url, tmp, lang):
    o = _opts(tmp) | {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt/srt/best",
    }
    try:
        with yt_dlp.YoutubeDL(o) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as e:
        log.warning("caption fetch failed: %s", e)
        return None
    subs = glob.glob(os.path.join(tmp, "*.vtt")) + glob.glob(os.path.join(tmp, "*.srt"))
    return _parse_subs(subs[0]) if subs else None


def _parse_subs(path):
    lines, last = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = re.sub(r"<[^>]+>", "", raw).strip()
            if (not line or "-->" in line or line.isdigit()
                    or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE"))):
                continue
            if line != last:  # auto-captions repeat lines as they scroll
                lines.append(line)
                last = line
    return "\n".join(lines) or None


def _fetch_audio(url, tmp, report):
    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                report("Downloading audio",
                       round(d.get("downloaded_bytes", 0) / total * 25))

    o = _opts(tmp) | {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(o) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as e:
        log.warning("audio fetch failed: %s", e)
        raise TranscriptError(_friendly(e)) from e
    wavs = glob.glob(os.path.join(tmp, "*.wav"))
    if not wavs:
        raise TranscriptError("Couldn't extract audio from this post.")
    return wavs[0]


def _whisper(wav, tmp, duration, report):
    out = os.path.join(tmp, "transcript")
    cmd = [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav, "-l", "auto",
           "-t", "4", "--print-progress", "-otxt", "-of", out]
    log.info("whisper start (%ss of audio)", duration)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    # Watchdog: GPU transcription runs ~60x realtime; duration/2 + 120s is
    # generous. A hung process gets killed instead of blocking jobs forever.
    timer = threading.Timer(max(180, duration / 2 + 120), proc.kill)
    timer.start()
    try:
        for line in proc.stderr:  # "...progress = 15%"
            m = re.search(r"progress\s*=\s*(\d+)%", line)
            if m:
                report("Transcribing", round(25 + int(m.group(1)) * 0.75))
        if proc.wait() != 0:
            log.error("whisper exited %s", proc.returncode)
            raise TranscriptError("Transcription failed or timed out.")
    finally:
        timer.cancel()
    try:
        with open(out + ".txt", encoding="utf-8") as f:
            return f.read().strip() or None
    except FileNotFoundError:
        raise TranscriptError("Transcription produced no output.") from None


def _friendly(e):
    msg = str(e)
    low = msg.lower()
    if "unsupported url" in low:
        return "That link isn't supported — try the direct link to the post."
    if "login" in low or "cookies" in low or "private" in low or "authentication" in low:
        return ("This post needs a login to access (common on Instagram). "
                "Export your browser cookies to grab/cookies.txt to fix this.")
    if "429" in msg or "rate" in low:
        return "The platform is rate-limiting us — wait a minute and retry."
    if "geo" in low or "country" in low:
        return "This post isn't available from your region."
    return msg.split(";")[0][:300]
