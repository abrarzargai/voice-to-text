#!/usr/bin/env python3
"""Web frontend for transcribe.py: upload a recording, transcribe it, write meeting minutes.

Run:   ~/venv/bin/python app.py      then open http://127.0.0.1:5000

Everything is kept in data/: settings and records (with every transcript and minutes version)
in data/db.json, and the uploaded audio in data/audio/ so a record can be transcribed again.
Meeting minutes are written (always in English) by a local LLM served by Ollama.
"""
import json
import os
import re
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import faster_whisper
from faster_whisper.utils import _MODELS as WHISPER_REPOS, download_model
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from huggingface_hub.constants import HF_HUB_CACHE

from catalog import DEFAULT_LLM, DEFAULT_WHISPER, LLM_CATALOG, WHISPER_BY_ID, WHISPER_CATALOG
from store import Store
from transcribe import MODELS, fmt_time, render
from whisper_worker import WhisperWorker, WorkerError

BASE = Path(__file__).resolve().parent
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

DEFAULT_OPTIONS = {"task": "transcribe", "language": "", "format": "timestamps", "device": "auto",
                   "beam_size": 5, "vad": True}
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", BASE / "data"))
store = Store(DATA_DIR, {
    "whisper_model": DEFAULT_WHISPER, "llm_model": DEFAULT_LLM, "auto_minutes": False, "theme": None,
    "options": DEFAULT_OPTIONS,
})
if store.fresh:  # first run with data/: bring in transcripts made by older versions of the app
    store.import_outputs(BASE / "outputs")

MINUTES_LANGUAGES = {"en": "English"}  # languages the minutes can be written in (more later)
MAX_MINUTES_TOKENS = 2048  # far more than real minutes need; stops a model that never ends
TIMESTAMP_RE = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]\s*")

MINUTES_PROMPT = """You are an expert meeting secretary. You will receive the transcript of a \
meeting or voice note. The transcript may be in any language (for example Hindi, Urdu or a mix \
with English) and may contain speech-recognition errors.

Write professional minutes of the meeting. ALWAYS write the minutes in English, translating \
anything said in other languages. Use only information from the transcript; never invent names, \
dates, numbers or decisions. If something is not mentioned, write "Not mentioned".

Use exactly this Markdown structure:

# Minutes of Meeting: <short descriptive title>

**Date:** ...
**Participants:** ...

## Summary
<3-5 sentence overview>

## Key Discussion Points
- ...

## Decisions Made
- ...

## Action Items
| # | Task | Owner | Deadline |
|---|------|-------|----------|
| 1 | ... | ... | ... |

## Open Questions / Next Steps
- ...

Output only the minutes, with no preamble."""

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3  # 2 GB

jobs = {}
jobs_lock = threading.Lock()
# One heavy job at a time: Whisper and the LLM both want all CPU cores.
work_lock = threading.Lock()
whisper = WhisperWorker()  # separate process, so Stop can end a transcription instantly


def update(job_id, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)


downloads = {}  # "whisper:tiny" / "llm:qwen2.5:0.5b" -> {"status", "progress", "message", ...}
streams = {}  # task id -> socket of an open Ollama stream, shut down to stop it immediately
ACTIVE = {"queued", "loading", "transcribing", "generating", "downloading"}


class Cancelled(Exception):
    """Raised inside a worker when the user stops its task."""


class RecordDeleted(Exception):
    """The record a job was working for was deleted before the job finished."""


def cancel_requested(task_id):
    with jobs_lock:
        if task_id.startswith("dl:"):
            return downloads.get(task_id[3:], {}).get("cancel", False)
        return jobs.get(task_id, {}).get("cancel", False)


def check_cancel(task_id):
    if task_id and cancel_requested(task_id):
        raise Cancelled()


@contextmanager
def work_slot(job_id):
    """Take the single work slot, giving up if the job is stopped while it waits in the queue."""
    while not work_lock.acquire(timeout=0.5):
        check_cancel(job_id)
    try:
        check_cancel(job_id)
        yield
    finally:
        work_lock.release()


@contextmanager
def track_stream(task_id, resp):
    """Remember the response socket so cancel_task() can interrupt a blocking read."""
    sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
    if task_id and sock:
        with jobs_lock:
            streams[task_id] = sock
    try:
        yield
    finally:
        with jobs_lock:
            streams.pop(task_id, None)


def job_failed(job_id, e):
    if isinstance(e, RecordDeleted):
        update(job_id, status="cancelled", message="The record was deleted", finished=time.time())
    elif cancel_requested(job_id):
        update(job_id, status="cancelled", message="Stopped by user", finished=time.time())
    elif isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError):
        update(job_id, status="error", message=f"Cannot reach Ollama at {OLLAMA_URL}: {e.reason}",
               finished=time.time())
    else:
        message = str(e) if isinstance(e, WorkerError) else f"{type(e).__name__}: {e}"
        update(job_id, status="error", message=message, finished=time.time())


def active_jobs_full():
    with jobs_lock:
        return [dict(j) for j in jobs.values() if j["status"] in ACTIVE]


def active_jobs(record_id=None):
    with jobs_lock:
        return [{"id": j["id"], "kind": j["kind"]} for j in jobs.values()
                if j["status"] in ACTIVE and (record_id is None or j["record"] == record_id)]


# ---------- models ----------
def set_download(key, **fields):
    with jobs_lock:
        downloads.setdefault(key, {}).update(fields)


def whisper_installed(name):
    try:
        download_model(name, local_files_only=True)
        return True
    except Exception:
        return False


def whisper_repo(name):
    """Hugging Face repo for a Whisper model id (a faster-whisper size name or a repo id)."""
    if name in WHISPER_BY_ID:
        return WHISPER_BY_ID[name]["repo"]
    return WHISPER_REPOS.get(name, name)


def whisper_repo_dir(name):
    return Path(HF_HUB_CACHE) / ("models--" + whisper_repo(name).replace("/", "--"))


def tree_size(path):
    """Bytes of regular files under path (symlinks not followed)."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def whisper_disk(name):
    # Snapshot files may be symlinks into a shared blob store, so follow and de-duplicate them.
    files = {f.resolve() for f in (whisper_repo_dir(name) / "snapshots").rglob("*") if f.is_file()}
    return sum(f.stat().st_size for f in files)


def ollama_models():
    """{name: tag info} for models already pulled into Ollama, or None if Ollama is not running."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return {m["name"]: m for m in json.load(r).get("models", [])}
    except Exception:
        return None


def system_info():
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            mem[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    disk = shutil.disk_usage(Path.home())
    version = None
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=3) as r:
            version = json.load(r).get("version")
    except Exception:
        pass
    return {"ram_total": mem.get("MemTotal"), "ram_available": mem.get("MemAvailable"),
            "disk_free": disk.free, "cpus": os.cpu_count(), "whisper_cache": HF_HUB_CACHE,
            "whisper_version": faster_whisper.__version__, "ollama_url": OLLAMA_URL,
            "ollama_version": version}


def ollama_pull(model, on_progress, task_id=None):
    """Pull a model into Ollama, calling on_progress(fraction_or_None, status_text)."""
    req = urllib.request.Request(f"{OLLAMA_URL}/api/pull",
                                 data=json.dumps({"model": model, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp, track_stream(task_id, resp):
        for line in resp:
            check_cancel(task_id)
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            total, done = chunk.get("total"), chunk.get("completed")
            on_progress(done / total if total and done else None, chunk.get("status", ""))
    check_cancel(task_id)  # a shut-down stream ends like a normal one


def run_download(kind, model_id):
    key = f"{kind}:{model_id}"
    try:
        if kind == "whisper":
            # huggingface_hub gives no progress callback; watch the cache grow instead. Files land in
            # the repo folder or in the shared blob store, depending on the huggingface_hub version.
            expected = WHISPER_BY_ID.get(model_id, {}).get("bytes")
            dirs = [whisper_repo_dir(model_id), Path(HF_HUB_CACHE) / "blobs"]
            baseline = sum(map(tree_size, dirs))
            finished = threading.Event()

            def watch():
                while not finished.wait(1):
                    size = max(sum(map(tree_size, dirs)) - baseline, 0)
                    set_download(key, progress=min(size / expected, 0.99) if expected else None,
                                 message=f"Downloading · {size / 1e6:.0f} MB")

            threading.Thread(target=watch, daemon=True).start()
            try:
                download_model(model_id)
            finally:
                finished.set()
        else:
            ollama_pull(model_id, lambda frac, text: set_download(key, progress=frac, message=text),
                        task_id="dl:" + key)
        set_download(key, status="done", progress=1.0, message="Installed", finished=time.time())
    except Exception as e:
        if cancel_requested("dl:" + key):
            set_download(key, status="cancelled", message="Stopped by user", finished=time.time())
        else:
            set_download(key, status="error", message=f"{type(e).__name__}: {e}", finished=time.time())


# ---------- jobs ----------
def parse_options(src):
    """Validated transcription options from a form / JSON dict (missing keys fall back to defaults)."""
    get = lambda k: src.get(k, DEFAULT_OPTIONS[k])  # noqa: E731
    vad = get("vad")
    try:
        beam = max(1, min(int(get("beam_size")), 10))
    except (TypeError, ValueError):
        beam = 5
    return {
        "task": "translate" if get("task") == "translate" else "transcribe",
        "language": get("language") if re.fullmatch(r"[a-z]{2,3}", str(get("language") or "")) else "",
        "format": get("format") if get("format") in ("txt", "timestamps", "srt") else "timestamps",
        "device": get("device") if get("device") in ("auto", "cpu", "cuda") else "auto",
        "beam_size": beam,
        "vad": vad if isinstance(vad, bool) else str(vad) == "true",
    }


def transcript_for_llm(version):
    """Plain transcript text for the LLM: no SRT numbers/timecodes, no "[mm:ss]" prefixes, and no
    lines without words. Small models tend to copy timestamps endlessly instead of writing minutes."""
    lines = []
    for line in version["text"].splitlines():
        line = TIMESTAMP_RE.sub("", line.strip())
        if version.get("format") == "srt" and (line.isdigit() or "-->" in line):
            continue
        if re.search(r"\w", line):
            lines.append(line)
    return "\n".join(lines)


def looks_stuck(text):
    """True when the end of the output is highly repetitive (e.g. "[08:40] [08:41] ..." forever).
    Real minutes compress to ~0.45 of their size; such loops to ~0.2 or less."""
    tail = text[-1500:].encode()
    return len(text) >= 1500 and len(zlib.compress(tail)) / len(tail) < 0.25


def ollama_chat(model, messages, num_ctx):
    """Open a streaming /api/chat request. Thinking is turned off where the model supports it."""
    body = {"model": model, "messages": messages, "stream": True, "think": False,
            "options": {"num_ctx": num_ctx, "temperature": 0.2, "repeat_penalty": 1.1,
                        "num_predict": MAX_MINUTES_TOKENS}}
    for attempt in range(2):
        req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            if attempt == 0 and e.code == 400:  # model without a "think" switch
                body.pop("think")
                continue
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"Ollama error {e.code}: {detail}") from None


def new_job(kind, record_id, **fields):
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "kind": kind, "record": record_id, "status": "queued",
                        "message": "Queued", "progress": 0.0, "segments": [], "output": None,
                        "created": time.time(), **fields}
    return job_id


def run_transcription(job_id, record_id, opts):
    try:
        rec = store.get(record_id)
        if rec is None:
            raise RecordDeleted()
        if not rec.get("audio"):
            raise RuntimeError("This record has no audio file to transcribe")
        path = store.audio_dir / rec["audio"]["file"]
        update(job_id, status="queued", message="Waiting for previous job to finish...")
        with work_slot(job_id):
            update(job_id, status="loading",
                   message=f"Loading model '{opts['model']}' (downloads on first use)...")
            worker_opts = {**opts, "language": opts["language"] or None}
            result, info = [], {}
            for msg in whisper.run(path, worker_opts, lambda: check_cancel(job_id)):
                if msg[0] == "loaded":
                    update(job_id, status="transcribing", message="Detecting language...")
                elif msg[0] == "info":
                    _, language, probability, duration = msg
                    info = {"detected_language": language, "duration": duration}
                    update(job_id, language=language, language_probability=round(probability, 3),
                           duration=duration,
                           message=f"Language: {language} ({probability:.0%}), duration {fmt_time(duration)}")
                else:
                    _, start, end, text = msg
                    if not re.search(r"\w", text):  # "." or "..." that Whisper produces for silence
                        continue
                    result.append(SimpleNamespace(start=start, end=end, text=text))
                    with jobs_lock:
                        job = jobs[job_id]
                        job["progress"] = min(end / info["duration"], 1.0) if info.get("duration") else 0
                        job["segments"].append({"start": fmt_time(start), "text": text.strip()})

        if not result:  # don't save an empty version
            raise RuntimeError(f"Whisper '{opts['model']}' didn't recognize any speech. Try a larger model, "
                               "set the spoken language, or turn off Skip silence.")
        version = store.add_version(record_id, "transcripts", {
            "text": render(result, opts["format"]), "model": opts["model"], "task": opts["task"],
            "language": opts["language"], "format": opts["format"], "segments": len(result), **info,
        })
        if version is None:
            raise RecordDeleted()
        update(job_id, status="done", progress=1.0, finished=time.time(),
               output={"record": record_id, "kind": "transcripts", "version": version["id"], "n": version["n"]},
               message=f"Saved as transcript v{version['n']}")
    except Exception as e:  # surface any failure (or a stop request) to the UI
        job_failed(job_id, e)


def run_minutes(job_id, record_id, transcript_id, model, language):
    try:
        kind, source = store.find_version(record_id, transcript_id)
        if kind != "transcripts":
            raise RecordDeleted() if store.get(record_id) is None else RuntimeError("Transcript version not found")
        text = transcript_for_llm(source)
        if not text:
            raise ValueError("The transcript is empty")
        update(job_id, status="queued", message="Waiting for previous job to finish...")
        with work_slot(job_id):
            installed = ollama_models()
            if installed is None:
                raise RuntimeError(f"Ollama is not running at {OLLAMA_URL}. Start it with `ollama serve`.")
            if model not in installed:
                update(job_id, status="loading", message=f"Downloading '{model}' (first use)...")

                def pulled(frac, status):
                    update(job_id, status="loading", progress=frac or 0,
                           message=f"Downloading '{model}': {status}" + (f" {frac:.0%}" if frac else ""))
                ollama_pull(model, pulled, task_id=job_id)
            update(job_id, status="loading", message=f"Loading '{model}' and reading the transcript...")
            # ~3 chars per token for the transcript, plus room for the prompt and the answer.
            num_ctx = min(max(8192, len(text) // 3 + 4096), 32768)
            name = store.name_of(record_id) or "recording"
            messages = [{"role": "system", "content": MINUTES_PROMPT},
                        {"role": "user", "content": f"Transcript ({name}):\n\n{text}"}]
            gen_started = None
            with ollama_chat(model, messages, num_ctx) as resp, track_stream(job_id, resp):
                for line in resp:
                    check_cancel(job_id)
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                    msg = chunk.get("message") or {}
                    if msg.get("content"):
                        gen_started = gen_started or time.time()
                        with jobs_lock:
                            job = jobs[job_id]
                            job["text"] += msg["content"]
                            job["tokens"] += 1
                            spent = time.time() - gen_started
                            job["speed"] = job["tokens"] / spent if spent > 0 else 0
                            job["status"] = "generating"
                            job["message"] = f"Writing minutes · {job['tokens']} tokens"
                            written, tokens = job["text"], job["tokens"]
                        if tokens % 50 == 0 and looks_stuck(written):
                            raise RuntimeError(
                                f"'{model}' got stuck repeating itself instead of writing minutes. "
                                "Try a larger minutes model (e.g. Qwen 2.5 1.5B) in Settings.")
                    elif msg.get("thinking"):
                        update(job_id, status="generating", message="Model is thinking...")
                    if chunk.get("done") and chunk.get("done_reason") == "length":
                        raise RuntimeError(
                            f"'{model}' didn't finish within {MAX_MINUTES_TOKENS} tokens (it was probably "
                            "repeating itself). Try a larger minutes model in Settings.")
            check_cancel(job_id)

        with jobs_lock:
            minutes, tokens = jobs[job_id]["text"], jobs[job_id]["tokens"]
        minutes = re.sub(r"<think>.*?</think>", "", minutes, flags=re.S).strip()
        minutes = re.sub(r"^```(?:markdown|md)?\s*\n(.*)\n```$", r"\1", minutes, flags=re.S).strip()
        # small models sometimes echo the prompt's closing instruction
        minutes = re.sub(r"\n*Output only the minutes, with no preamble\.?\s*$", "", minutes).strip()
        if not minutes:
            raise RuntimeError("The model returned no text")
        version = store.add_version(record_id, "minutes", {
            "text": minutes + "\n", "model": model, "source": source["id"], "source_n": source["n"],
            "tokens": tokens, "language": language,
        })
        if version is None:
            raise RecordDeleted()
        update(job_id, status="done", progress=1.0, finished=time.time(),
               output={"record": record_id, "kind": "minutes", "version": version["id"], "n": version["n"]},
               message=f"Saved as minutes v{version['n']}")
    except Exception as e:
        job_failed(job_id, e)


def start_transcription(record_id, opts):
    job_id = new_job("transcribe", record_id, options=opts)
    threading.Thread(target=run_transcription, args=(job_id, record_id, opts), daemon=True).start()
    return job_id


def start_minutes(record_id, transcript_id, model, language):
    job_id = new_job("minutes", record_id, model=model, source=transcript_id, text="", tokens=0, speed=0)
    threading.Thread(target=run_minutes, args=(job_id, record_id, transcript_id, model, language),
                     daemon=True).start()
    return job_id


def whisper_model_from(value):
    model = value or store.settings()["whisper_model"]
    if model not in WHISPER_BY_ID and model not in MODELS:
        abort(Response(json.dumps({"error": f"Unknown model {model}"}), 400, mimetype="application/json"))
    return model


# ---------- pages & settings ----------
@app.get("/")
def index():
    return send_from_directory(BASE / "static", "index.html")


@app.get("/api/settings")
def get_settings():
    return jsonify(store.settings())


@app.put("/api/settings")
def put_settings():
    data = request.get_json(silent=True) or {}
    patch = {}
    for key in ("whisper_model", "llm_model"):
        if isinstance(data.get(key), str) and data[key]:
            patch[key] = data[key]
    if isinstance(data.get("auto_minutes"), bool):
        patch["auto_minutes"] = data["auto_minutes"]
    if data.get("theme") in (None, "light", "dark") and "theme" in data:
        patch["theme"] = data["theme"]
    if isinstance(data.get("options"), dict):
        patch["options"] = parse_options({**store.settings()["options"], **data["options"]})
    return jsonify(store.update_settings(patch))


# ---------- records ----------
def record_or_404(rid):
    rec = store.get(rid)
    if rec is None:
        abort(Response(json.dumps({"error": "Record not found"}), 404, mimetype="application/json"))
    return rec


@app.get("/api/records")
def list_records():
    busy = {j["record"] for j in active_jobs_full()}
    return jsonify([r | {"busy": r["id"] in busy} for r in store.summaries()])


@app.post("/api/records")
def create_record():
    """Upload a recording: creates a record, keeps the audio, and starts the first transcription."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file uploaded"), 400
    model = whisper_model_from(request.form.get("model"))
    opts = parse_options(request.form) | {"model": model}

    original = f.filename.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(original).suffix.lower()
    suffix = suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else ""
    audio_file = f"{uuid.uuid4().hex[:12]}{suffix}"
    path = store.audio_dir / audio_file
    f.save(path)
    rec = store.create(request.form.get("name") or Path(original).stem,
                       audio={"file": audio_file, "name": original, "size": path.stat().st_size})
    return jsonify(record=rec["id"], name=rec["name"], job=start_transcription(rec["id"], opts))


@app.get("/api/records/<rid>")
def get_record(rid):
    rec = record_or_404(rid)
    return jsonify(rec | {"active_jobs": active_jobs(rid)})


@app.patch("/api/records/<rid>")
def rename_record(rid):
    record_or_404(rid)
    try:
        name = store.rename(rid, (request.get_json(silent=True) or {}).get("name", ""))
    except ValueError as e:
        return jsonify(error=str(e)), 409
    return jsonify(name=name)


@app.delete("/api/records/<rid>")
def delete_record(rid):
    record_or_404(rid)
    for job in active_jobs(rid):  # stop work for this record before its audio disappears
        with jobs_lock:
            jobs[job["id"]]["cancel"] = True
        _interrupt(job["id"])
    audio = store.delete(rid)
    if audio:
        (store.audio_dir / audio).unlink(missing_ok=True)
    return jsonify(ok=True)


@app.get("/api/records/<rid>/audio")
def record_audio(rid):
    rec = record_or_404(rid)
    if not rec.get("audio"):
        abort(404)
    return send_file(store.audio_dir / rec["audio"]["file"], download_name=rec["audio"]["name"],
                     conditional=True)


@app.post("/api/records/<rid>/transcribe")
def retranscribe(rid):
    rec = record_or_404(rid)
    if not rec.get("audio"):
        return jsonify(error="This record has no audio file, so it can't be transcribed again"), 400
    data = request.get_json(silent=True) or {}
    opts = parse_options(data) | {"model": whisper_model_from(data.get("model"))}
    return jsonify(job=start_transcription(rid, opts))


@app.post("/api/records/<rid>/minutes")
def generate_minutes(rid):
    rec = record_or_404(rid)
    data = request.get_json(silent=True) or {}
    if not rec["transcripts"]:
        return jsonify(error="Transcribe the recording first"), 400
    transcript_id = data.get("transcript") or rec["transcripts"][-1]["id"]
    if not any(v["id"] == transcript_id for v in rec["transcripts"]):
        return jsonify(error="Transcript version not found"), 404
    model = data.get("model") or store.settings()["llm_model"]
    if not re.fullmatch(r"[\w.\-/:]+", model):
        return jsonify(error="Invalid model name"), 400
    if any(j["kind"] == "minutes" for j in active_jobs(rid)):
        return jsonify(error="Minutes are already being generated for this recording"), 409
    language = data.get("language") or "en"
    if language not in MINUTES_LANGUAGES:
        return jsonify(error=f"Minutes can't be written in '{language}' yet"), 400
    return jsonify(job=start_minutes(rid, transcript_id, model, language))


@app.put("/api/records/<rid>/versions/<vid>")
def edit_version(rid, vid):
    """Save an edited transcript or minutes version (kept in place, marked as edited)."""
    record_or_404(rid)
    text = (request.get_json(silent=True) or {}).get("text")
    if not isinstance(text, str):
        return jsonify(error="Missing text"), 400
    kind, _ = store.find_version(rid, vid)
    if kind is None:
        return jsonify(error="Version not found"), 404
    return jsonify(store.edit_version(rid, vid, text.rstrip() + "\n"))


@app.delete("/api/records/<rid>/versions/<vid>")
def delete_version(rid, vid):
    record_or_404(rid)
    try:
        store.delete_version(rid, vid)
    except KeyError:
        return jsonify(error="Version not found"), 404
    return jsonify(ok=True)


@app.get("/api/records/<rid>/versions/<vid>/download")
def download_version(rid, vid):
    rec = record_or_404(rid)
    kind, v = store.find_version(rid, vid)
    if v is None:
        abort(404)
    ext = "md" if kind == "minutes" else "srt" if v.get("format") == "srt" else "txt"
    label = "minutes" if kind == "minutes" else "transcript"
    filename = f"{rec['name']} - {label} v{v['n']}.{ext}"
    return Response(v["text"], mimetype="text/plain; charset=utf-8", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"})


# ---------- models ----------
@app.get("/api/models")
def list_models():
    """Model catalog for Settings: the library, any other installed models, and system info."""
    with jobs_lock:
        dls = {k: dict(v) for k, v in downloads.items()}

    def entry(kind, choice, disk=None, extra=False):
        installed = disk is not None
        dl = dls.get(f"{kind}:{choice['id']}", {})
        if installed and dl.get("status") != "downloading":
            dl = {}
        return choice | {"installed": installed, "disk": disk, "extra": extra, "download": dl}

    whisper_models = [entry("whisper", c, whisper_disk(c["id"]) if whisper_installed(c["id"]) else None)
                      for c in WHISPER_CATALOG]

    pulled = ollama_models()
    llm = [entry("llm", c, pulled[c["id"]]["size"] if pulled and c["id"] in pulled else None)
           for c in LLM_CATALOG]
    curated = {c["id"] for c in LLM_CATALOG}
    for name, info in (pulled or {}).items():
        if name in curated:
            continue
        details = info.get("details") or {}
        spec = " · ".join(filter(None, [details.get("parameter_size"), details.get("quantization_level")]))
        llm.append(entry("llm", {
            "id": name, "label": name, "bytes": info["size"],
            "ram_gb": round(info["size"] / 1e9 + 1, 1), "speed": "Depends on size", "quality": spec or "—",
            "langs": "—", "recommended": False,
            "good_for": "Already in Ollama on this machine. Models above ~4B parameters will be slow "
                        "on CPU and need a lot of free RAM.",
        }, info["size"], extra=True))
    return jsonify(whisper=whisper_models, llm=llm, ollama=pulled is not None, system=system_info(),
                   checked=time.time(), default_whisper=DEFAULT_WHISPER, default_llm=DEFAULT_LLM)


@app.post("/api/models/download")
def start_download():
    data = request.get_json(silent=True) or {}
    kind, model_id = data.get("kind"), data.get("id") or ""
    if kind == "whisper" and model_id not in WHISPER_BY_ID and model_id not in WHISPER_REPOS:
        return jsonify(error=f"Unknown Whisper model {model_id}"), 400
    if kind == "llm":
        if ollama_models() is None:
            return jsonify(error=f"Ollama is not running at {OLLAMA_URL}"), 503
        if not re.fullmatch(r"[\w.\-/:]+", model_id):
            return jsonify(error="Invalid model name"), 400
    elif kind != "whisper":
        return jsonify(error="Unknown model kind"), 400

    key = f"{kind}:{model_id}"
    with jobs_lock:
        if downloads.get(key, {}).get("status") == "downloading":
            return jsonify(ok=True)
        downloads[key] = {"status": "downloading", "progress": 0.0, "message": "Starting download...",
                          "started": time.time(), "cancel": False}
    threading.Thread(target=run_download, args=(kind, model_id), daemon=True).start()
    return jsonify(ok=True)


# ---------- tasks ----------
@app.get("/api/tasks")
def list_tasks():
    """Running, queued and recently finished work, for the header's task menu."""
    now, out = time.time(), []
    with jobs_lock:
        job_list = [dict(j) for j in jobs.values()]
        dl_list = [(k, dict(d)) for k, d in downloads.items()]
    for j in job_list:
        active = j["status"] in ACTIVE
        if not active and now - j.get("finished", j["created"]) > 1800:
            continue
        out.append({
            "id": j["id"], "type": j["kind"], "record": j["record"],
            "title": store.name_of(j["record"]) or "(deleted record)", "status": j["status"],
            "detail": f"Whisper {j['options']['model']}" if j["kind"] == "transcribe" else j["model"],
            "message": j["message"], "progress": j.get("progress"), "active": active,
            "stopping": active and j.get("cancel", False), "cancellable": active,
            "started": j["created"], "finished": j.get("finished"), "output": j.get("output"),
        })
    for key, d in dl_list:
        active = d.get("status") == "downloading"
        if not active and now - d.get("finished", d.get("started", 0)) > 1800:
            continue
        kind, model_id = key.split(":", 1)
        out.append({
            "id": "dl:" + key, "type": "download", "record": None, "title": model_id,
            "detail": "Speech-to-text model" if kind == "whisper" else "Minutes model (Ollama)",
            "status": d.get("status"), "message": d.get("message", ""), "progress": d.get("progress"),
            "active": active, "stopping": active and d.get("cancel", False),
            # huggingface_hub offers no way to abort a running download
            "cancellable": active and kind == "llm",
            "started": d.get("started"), "finished": d.get("finished"), "output": None,
        })
    out.sort(key=lambda t: (not t["active"], -(t["started"] or 0)))
    return jsonify(out)


def _interrupt(task_id):
    """Shut down an open Ollama stream so a read blocked on the prompt returns at once."""
    with jobs_lock:
        sock = streams.get(task_id)
    if sock:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


@app.post("/api/tasks/<path:task_id>/cancel")
def cancel_task(task_id):
    with jobs_lock:
        if task_id.startswith("dl:"):
            item = downloads.get(task_id[3:])
            if item and task_id.startswith("dl:whisper:"):
                return jsonify(error="Whisper downloads can't be stopped once started"), 409
            running = item and item.get("status") == "downloading"
        else:
            item = jobs.get(task_id)
            running = item and item["status"] in ACTIVE
        if not running:
            return jsonify(error="This task is not running"), 409
        item["cancel"] = True
        item["message"] = "Stopping..."
    _interrupt(task_id)
    return jsonify(ok=True)


@app.post("/api/tasks/clear")
def clear_tasks():
    with jobs_lock:
        for job_id in [k for k, j in jobs.items() if j["status"] not in ACTIVE]:
            del jobs[job_id]
        for key in [k for k, d in downloads.items() if d.get("status") != "downloading"]:
            del downloads[key]
    return jsonify(ok=True)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    since = request.args.get("since", 0, type=int)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            abort(404)
        data = {k: v for k, v in job.items() if k != "segments"}
        data["segments"] = job["segments"][since:]
        data["segment_count"] = len(job["segments"])
    data["record_name"] = store.name_of(data["record"])
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
