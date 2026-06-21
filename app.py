"""Grab — paste a social media link, get video / audio / transcript.

Run:  .venv/bin/python app.py
Then open http://localhost:5001 (or http://<your-lan-ip>:5001 from your phone).
"""

import glob
import os

import certifi

# macOS Python ships without CA certs wired up; yt-dlp needs them for HTTPS.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import logging
import re
import secrets
import shutil
import socket
import tempfile
import threading
import time

from flask import Flask, jsonify, request, send_file, send_from_directory

import yt_dlp

import transcript

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("grab")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Where ffmpeg/ffprobe actually live (Homebrew puts them in /usr/local/bin or
# /opt/homebrew/bin, NOT /usr/bin). Detect instead of hardcoding so downloads
# don't fail with "ffmpeg not found" on machines with a different layout.
_ffmpeg = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
FFMPEG_DIR = os.path.dirname(_ffmpeg)

app = Flask(__name__, static_folder="static")


@app.get("/")
def index():
    resp = send_from_directory(app.static_folder, "index.html")
    # Phones cache aggressively; a stale page can't talk to a newer API.
    resp.headers["Cache-Control"] = "no-store"
    return resp


def ydl_base(tmp):
    o = {
        "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "socket_timeout": 30,
        "ffmpeg_location": FFMPEG_DIR,
        # Use mobile/TV clients to avoid bot detection on cloud server IPs
        "extractor_args": {"youtube": {"player_client": ["ios", "android", "tv_embedded"]}},
    }
    if os.path.exists(transcript.COOKIES):  # helps with login-walled posts
        o["cookiefile"] = transcript.COOKIES
    return o


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


# a job "running" this long is treated as dead — generous so long videos
# (multi-hour transcription / large downloads) finish. Override w/ env.
STUCK_AFTER = int(os.environ.get("GRAB_STUCK_HOURS", "6")) * 3600


def prune():
    now = time.time()
    with JOBS_LOCK:
        for k in [k for k, j in JOBS.items() if now - j["at"] > JOB_KEEP_SECONDS]:
            del JOBS[k]
        for k, j in JOBS.items():
            if j["status"] == "running" and now - j.get("started", now) > STUCK_AFTER:
                log.error("job %s stuck >%ss, marking failed", k, STUCK_AFTER)
                j.update(status="error", at=now,
                         error="This task got stuck and was cancelled — try again.")
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
                return jsonify({k: v for k, v in job.items()
                                if k not in ("at", "started")})
        JOBS[key] = {"status": "running", "progress": None,
                     "stage": "Starting…", "at": time.time(),
                     "started": time.time()}

    worker = script_worker if kind == "script" else download_worker
    threading.Thread(target=worker, args=(key, url, kind), daemon=True).start()
    with JOBS_LOCK:
        return jsonify({k: v for k, v in JOBS[key].items()
                        if k not in ("at", "started")})


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
        log.info("download done %s: %s", kind, READY[token][1])
        set_job(key, status="done", stage="Ready", progress=100,
                token=token, size=os.path.getsize(path))
    except Exception as e:
        log.warning("download failed %s %s: %s", kind, url, e)
        shutil.rmtree(tmp, ignore_errors=True)
        set_job(key, status="error", error=friendly(e))


def script_worker(key, url, kind):
    """Run the transcript pipeline (transcript.py) as a polled job."""
    tmp = tempfile.mkdtemp(prefix="grab_")

    def report(stage, progress):
        set_job(key, stage=stage, progress=progress)

    # At most two transcripts at once; extra requests queue instead of failing.
    queued = not SCRIPT_SLOTS.acquire(blocking=False)
    if queued:
        report("Waiting for a free slot…", None)
        SCRIPT_SLOTS.acquire()
    try:
        text, source = transcript.get_transcript(url, tmp, report)
        set_job(key, status="done", stage="Ready", progress=100,
                text=text, source=source)
    except transcript.TranscriptError as e:
        set_job(key, status="error", error=str(e))
    except Exception as e:
        log.exception("script job crashed: %s", url)
        set_job(key, status="error", error="Unexpected error: " + str(e)[:200])
    finally:
        SCRIPT_SLOTS.release()
        shutil.rmtree(tmp, ignore_errors=True)


SCRIPT_SLOTS = threading.Semaphore(2)


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
    port = int(os.environ.get("PORT", 5001))
    if port == 5001:
        print(f"\n  On this Mac:    http://localhost:5001")
        print(f"  On your phone:  http://{lan_ip()}:5001  (same Wi-Fi)\n")
    app.run(host="0.0.0.0", port=port, threaded=True)
