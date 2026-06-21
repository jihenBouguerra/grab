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
import json
import logging
import os
import re
import ssl
import subprocess
import threading
import time
import urllib.request

import certifi
import yt_dlp

_SSL = ssl.create_default_context(cafile=certifi.where())

log = logging.getLogger("grab.transcript")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL = os.path.join(APP_DIR, "models", "ggml-base.bin")
_local = os.path.join(APP_DIR, "bin", "whisper-cli")
WHISPER_BIN = _local if os.path.exists(_local) else None
COOKIES = os.path.join(APP_DIR, "cookies.txt")  # optional, helps Instagram

CAPTION_LANGS = ["en", "de", "ar", "fr"]   # preference order
# 0 = no limit (transcribe any length). Override with GRAB_MAX_MINUTES.
MAX_DURATION = int(os.environ.get("GRAB_MAX_MINUTES", "0")) * 60


class TranscriptError(Exception):
    """User-readable failure."""


def _opts(tmp=None):
    o = {"quiet": True, "noprogress": True, "no_warnings": True,
         "noplaylist": True, "socket_timeout": 30,
         "extractor_args": {"youtube": {"player_client": ["ios", "android", "tv_embedded"]}}}
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
        text = _fetch_captions(info, lang, report)
        if text:
            log.info("script done via captions[%s]: %d chars", lang, len(text))
            return text, "captions"
        log.warning("captions[%s] advertised but unusable, falling back", lang)

    if not WHISPER_BIN or not os.path.exists(WHISPER_MODEL):
        raise TranscriptError("This video has no captions, and local "
                              "transcription isn't set up (run ./setup.sh).")
    if MAX_DURATION and duration and duration > MAX_DURATION:
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


def _caption_tracks(info):
    """All caption tracks from the probe, manual subtitles winning over auto."""
    return {**(info.get("automatic_captions") or {}),
            **(info.get("subtitles") or {})}


def _best_caption_lang(info):
    """Prefer the video's ORIGINAL spoken language. Asking for a language the
    video wasn't recorded in (e.g. 'en' on an Arabic video) forces YouTube to
    machine-translate on the fly — those requests get rate-limited (HTTP 429)
    and are slow. The native track is served instantly."""
    available = _caption_tracks(info)
    if not available:
        return None
    orig = info.get("language")
    # 1. original language (+ its '-orig' auto track), 2. any '-orig' track,
    # 3. the user's preference list, 4. anything that exists.
    wants = []
    if orig:
        wants += [orig, orig + "-orig"]
    wants += [c for c in available if c.endswith("-orig")]
    wants += CAPTION_LANGS
    wants += list(available)
    for want in wants:
        for code in available:  # 'en' should also match 'en-US' etc.
            if code == want or code.startswith(want + "-"):
                return code
    return None


def _fetch_captions(info, lang, report=None):
    """Download the caption track directly from the URL the probe already gave
    us — one clean request, no second yt-dlp extraction. json3 preferred."""
    entries = _caption_tracks(info).get(lang) or []
    pref = ["json3", "srv3", "vtt", "srt", "srv1", "srv2", "ttml"]
    entries = sorted(
        entries,
        key=lambda e: pref.index(e["ext"]) if e.get("ext") in pref else 99)
    for e in entries:
        url = e.get("url")
        if not url:
            continue
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=20, context=_SSL).read()
            except Exception as ex:
                if "429" in str(ex) and attempt == 0:
                    if report:
                        report("Captions busy, retrying…", None)
                    time.sleep(2)
                    continue
                log.warning("caption fetch failed (%s): %s", e.get("ext"), ex)
                break
            text = _parse_caption(data, e.get("ext"))
            if text:
                return text
            break  # got the file but it was empty — try next format
    return None


def _parse_caption(data, ext):
    if ext == "json3":
        try:
            events = json.loads(data).get("events", [])
        except Exception:
            return None
        out, last = [], None
        for ev in events:
            seg = "".join(s.get("utf8", "") for s in (ev.get("segs") or [])).strip()
            if seg and seg != last:
                out.append(seg)
                last = seg
        return "\n".join(out) or None
    # vtt / srt / ttml: strip tags + cue timing, dedupe scrolling repeats
    lines, last = [], None
    for raw in data.decode("utf-8", "replace").splitlines():
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
           "-t", str(os.environ.get("GRAB_WHISPER_THREADS", "8")),
           "--print-progress", "-otxt", "-of", out]
    log.info("whisper start (%ss of audio)", duration)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    # Watchdog: only kills a genuinely hung process. Allow up to ~2x realtime
    # + 5 min so even slow CPU transcription of a long video finishes.
    timer = threading.Timer(max(300, duration * 2 + 300), proc.kill)
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
    if "sign in to confirm" in low or ("bot" in low and "confirm" in low):
        return "YouTube is blocking this server's IP as a bot. Try a different video or use TikTok/Instagram instead."
    if "login" in low or "private" in low or "authentication" in low:
        return ("This post needs a login to access (common on Instagram). "
                "Export your browser cookies to grab/cookies.txt to fix this.")
    if "cookies" in low:
        return "YouTube is blocking requests from this server. Try a different platform or video."
    if "429" in msg or "rate" in low:
        return "The platform is rate-limiting us — wait a minute and retry."
    if "geo" in low or "country" in low:
        return "This post isn't available from your region."
    return msg.split(";")[0][:300]
