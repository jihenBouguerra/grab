"""Grab — paste a social media link, get video / audio / transcript.

Run:  .venv/bin/python app.py
Then open http://localhost:5001 (or http://<your-lan-ip>:5001 from your phone).
"""

import glob
import os

import certifi

# macOS Python ships without CA certs wired up; yt-dlp needs them for HTTPS.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time

from flask import Flask, jsonify, request, send_file, send_from_directory

import yt_dlp

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL = os.path.join(APP_DIR, "models", "ggml-base.bin")
# Prefer our native arm64+Metal build (GPU, ~65x realtime); the Homebrew one
# in /usr/local is an Intel binary that crawls under Rosetta on this Mac.
_local_whisper = os.path.join(APP_DIR, "bin", "whisper-cli")
WHISPER_BIN = (_local_whisper if os.path.exists(_local_whisper)
               else shutil.which("whisper-cli") or shutil.which("whisper-cpp"))
MAX_TRANSCRIBE_SECONDS = 20 * 60  # refuse to whisper-transcribe longer videos

app = Flask(__name__, static_folder="static")


@app.get("/")
def index():
    resp = send_from_directory(app.static_folder, "index.html")
    # Phones cache aggressively; a stale page can't talk to a newer API.
    resp.headers["Cache-Control"] = "no-store"
    return resp


def ydl_base(tmp):
    return {
        "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }


def safe_name(title, ext):
    name = re.sub(r"[^\w\s\-.]", "", title or "download").strip()[:80] or "download"
    return f"{name}.{ext}"


@app.post("/api/info")
def info():
    url = (request.json or {}).get("url", "").strip()
    if not url:
        return jsonify(error="No link given"), 400
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
            i = ydl.extract_info(url, download=False)
        if i.get("_type") == "playlist":
            i = (i.get("entries") or [{}])[0]
        return jsonify(
            title=i.get("title"),
            uploader=i.get("uploader") or i.get("channel"),
            duration=i.get("duration"),
            thumbnail=i.get("thumbnail"),
        )
    except Exception as e:
        return jsonify(error=friendly(e)), 422


# ---------------------------------------------------------------------------
# Jobs: every video / audio / script request runs as a background job the page
# polls. Phones lock their screen mid-task, which kills a plain HTTP request —
# polling re-attaches to the same job after a reload, and gives us a place to
# report progress from.
# ---------------------------------------------------------------------------
JOBS = {}  # "kind|url" -> {status, progress, stage, ..., at}
JOBS_LOCK = threading.Lock()
JOB_KEEP_SECONDS = 3600

# Finished downloads waiting for pickup: token -> (path, name, tmpdir, created)
READY = {}
READY_TTL = 600


def set_job(key, **kw):
    with JOBS_LOCK:
        if key in JOBS:
            JOBS[key].update(kw, at=time.time())


def prune():
    now = time.time()
    with JOBS_LOCK:
        for k in [k for k, j in JOBS.items() if now - j["at"] > JOB_KEEP_SECONDS]:
            del JOBS[k]
        for t in [t for t, v in READY.items() if now - v[3] > READY_TTL]:
            shutil.rmtree(READY.pop(t)[2], ignore_errors=True)


def progress_hook(key, lo, hi):
    """yt-dlp hook mapping download progress onto the [lo, hi] % range."""
    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                done = d.get("downloaded_bytes", 0) / total
                set_job(key, progress=round(lo + done * (hi - lo)))
        elif d.get("status") == "finished":
            set_job(key, progress=hi)
    return hook


@app.post("/api/job")
def job_api():
    prune()
    data = request.json or {}
    url = data.get("url", "").strip()
    kind = data.get("kind", "script")
    if not url:
        return jsonify(error="No link given"), 400
    if kind not in ("video", "audio", "script"):
        return jsonify(error="Unknown job kind"), 400

    key = f"{kind}|{url}"
    with JOBS_LOCK:
        job = JOBS.get(key)
        if job:
            # Re-clicking after a failure retries; polls just report state.
            # A finished download whose file already expired must also restart.
            stale = (job["status"] == "error"
                     or (job["status"] == "done" and "token" in job
                         and job["token"] not in READY))
            if stale and not data.get("poll"):
                del JOBS[key]
            else:
                return jsonify({k: v for k, v in job.items() if k != "at"})
        if kind == "script" and any(
                k.startswith("script|") and j["status"] == "running"
                for k, j in JOBS.items()):
            return jsonify(error="Another script is still being prepared — "
                                 "wait for it to finish."), 429
        JOBS[key] = {"status": "running", "progress": None,
                     "stage": "Starting…", "fast": bool(data.get("fast")),
                     "at": time.time()}

    worker = script_worker if kind == "script" else download_worker
    threading.Thread(target=worker, args=(key, url, kind), daemon=True).start()
    with JOBS_LOCK:
        return jsonify({k: v for k, v in JOBS[key].items() if k != "at"})


# Old page versions (cached on phones) still POST here — same job machinery.
@app.post("/api/script")
def script_compat():
    return job_api()


@app.post("/api/prepare")
def prepare_compat():
    """Synchronous download for stale cached pages: do the work inline."""
    data = request.json or {}
    url, kind = data.get("url", "").strip(), data.get("kind", "video")
    if not url:
        return jsonify(error="No link given"), 400
    key = f"{kind}|{url}"
    with JOBS_LOCK:
        JOBS.setdefault(key, {"status": "running", "progress": None,
                              "stage": "Starting…", "at": time.time()})
    download_worker(key, url, kind)
    with JOBS_LOCK:
        job = JOBS.get(key, {})
    if job.get("status") == "done":
        return jsonify(token=job["token"], size=job["size"])
    return jsonify(error=job.get("error", "Download failed")), 422


@app.get("/api/take/<token>")
def take(token):
    # Don't consume the token: download managers may re-request or use ranges.
    # Files are cleaned up by prune() after READY_TTL instead.
    with JOBS_LOCK:
        entry = READY.get(token)
    if not entry:
        return jsonify(error="This download expired — try again."), 404
    path, name, _, _ = entry
    return send_file(path, as_attachment=True, download_name=name, conditional=True)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def download_worker(key, url, kind):
    tmp = tempfile.mkdtemp(prefix="grab_")
    try:
        set_job(key, stage="Downloading", progress=0)
        opts = ydl_base(tmp) | {"progress_hooks": [progress_hook(key, 0, 90)]}
        if kind == "audio":
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ]
        else:
            opts["format"] = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"
            opts["merge_output_format"] = "mp4"

        with yt_dlp.YoutubeDL(opts) as ydl:
            i = ydl.extract_info(url, download=True)
        if i.get("_type") == "playlist":
            i = (i.get("entries") or [{}])[0]
        set_job(key, stage="Converting", progress=95)

        ext = "mp3" if kind == "audio" else "mp4"
        files = glob.glob(os.path.join(tmp, f"*.{ext}")) or glob.glob(os.path.join(tmp, "*"))
        if not files:
            raise RuntimeError("Download produced no file")
        path = max(files, key=os.path.getsize)

        token = secrets.token_urlsafe(16)
        with JOBS_LOCK:
            READY[token] = (path, safe_name(i.get("title"), ext), tmp, time.time())
        set_job(key, status="done", stage="Ready", progress=100,
                token=token, size=os.path.getsize(path))
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        set_job(key, status="error", error=friendly(e))


def script_worker(key, url, kind):
    tmp = tempfile.mkdtemp(prefix="grab_")
    try:
        set_job(key, stage="Checking captions", progress=None)
        text = captions_text(url, tmp)
        source = "captions"
        if not text:
            duration = video_duration(url)
            if duration and duration > MAX_TRANSCRIBE_SECONDS:
                raise RuntimeError(
                    f"This video is {duration // 60} min long and has no captions. "
                    f"Local transcription is capped at {MAX_TRANSCRIBE_SECONDS // 60} "
                    f"min to avoid overheating this Mac.")
            set_job(key, stage="Downloading audio", progress=0)
            with JOBS_LOCK:
                fast = JOBS.get(key, {}).get("fast", False)
            text = whisper_text(key, url, tmp, fast)
            source = "transcription"
        if not text:
            raise RuntimeError("No captions found and local transcription is not set "
                               "up (whisper-cli or its model file is missing).")
        set_job(key, status="done", stage="Ready", progress=100,
                text=text, source=source)
    except Exception as e:
        set_job(key, status="error", error=friendly(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Captions & transcription
# ---------------------------------------------------------------------------
# Caption languages we accept, in order of preference. Asking for "all" gets
# rate-limited (429) by YouTube and pulls auto-translated junk like "en-de".
CAPTION_LANGS = ["en", "de", "ar", "fr"]


def captions_text(url, tmp):
    """Try platform-provided captions/auto-captions first (free and instant)."""
    opts = ydl_base(tmp) | {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt/srt/best",
        # Exact codes only — patterns like "en.*" also match auto-translated
        # pairs ("en-de"), which triggers extra downloads and 429 rate limits.
        "subtitleslangs": CAPTION_LANGS + ["en-US", "en-GB", "de-DE", "fr-FR"],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception:
        return None
    subs = glob.glob(os.path.join(tmp, "*.vtt")) + glob.glob(os.path.join(tmp, "*.srt"))
    if not subs:
        return None

    def rank(path):  # files look like <id>.<lang>.vtt — prefer our language order
        lang = path.rsplit(".", 2)[-2].lower()
        for n, code in enumerate(CAPTION_LANGS):
            if lang == code or lang.startswith(code + "-"):
                return n
        return len(CAPTION_LANGS)

    return parse_subs(min(subs, key=rank))


def parse_subs(path):
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


def video_duration(url):
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
            i = ydl.extract_info(url, download=False)
        if i.get("_type") == "playlist":
            i = (i.get("entries") or [{}])[0]
        return i.get("duration")
    except Exception:
        return None


def whisper_text(key, url, tmp, fast=False):
    """Fallback: download the audio and transcribe locally with whisper.cpp.
    Audio download maps to 0–25% of the bar, transcription to 25–100%."""
    if not WHISPER_BIN or not os.path.exists(WHISPER_MODEL):
        return None
    opts = ydl_base(tmp) | {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "progress_hooks": [progress_hook(key, 0, 25)],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    wavs = glob.glob(os.path.join(tmp, "*.wav"))
    if not wavs:
        return None

    set_job(key, stage="Transcribing (fast)" if fast else "Transcribing", progress=25)
    out = os.path.join(tmp, "transcript")
    # Quiet mode: 2 threads at lowest priority — slow but the fan stays calm.
    # Fast mode (the ⚡ toggle): most cores, normal-ish priority — 3-4x faster.
    threads = max(2, (os.cpu_count() or 4) - 2) if fast else 2
    niceness = "5" if fast else "19"
    proc = subprocess.Popen(
        ["nice", "-n", niceness, WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wavs[0],
         "-l", "auto", "-t", str(threads), "--print-progress", "-otxt", "-of", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    for line in proc.stderr:  # "whisper_print_progress_callback: progress = 15%"
        m = re.search(r"progress\s*=\s*(\d+)%", line)
        if m:
            set_job(key, progress=round(25 + int(m.group(1)) * 0.75))
    if proc.wait(timeout=1800) != 0:
        raise RuntimeError("Transcription failed")
    with open(out + ".txt", encoding="utf-8") as f:
        return f.read().strip() or None


def friendly(e):
    msg = str(e)
    if "Unsupported URL" in msg:
        return "That link isn't supported. Try the direct link to the post/video."
    if "login" in msg.lower() or "private" in msg.lower() or "rate-limit" in msg.lower():
        return "This post seems private or the platform is blocking anonymous access."
    return msg.split(";")[0][:300]


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


if __name__ == "__main__":
    print(f"\n  On this Mac:    http://localhost:5001")
    print(f"  On your phone:  http://{lan_ip()}:5001  (same Wi-Fi)\n")
    app.run(host="0.0.0.0", port=5001, threaded=True)
