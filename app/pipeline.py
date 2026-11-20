"""Acquire -> probe -> (transcribe || scan video & audio) -> analyze -> AI review / translate. One job at a time."""
import json
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import ai
from .analysis import export, text_stats

MODELS = ["large-v3", "large-v3-turbo", "distil-large-v3.5", "medium", "small", "base", "tiny"]
_model_lock = threading.Lock()
_model: dict = {}  # ponytail: one Whisper model resident at a time, VRAM is the limit
_cuda_broken = False  # set once CUDA libraries fail to load; the session then stays on CPU


def _register_cuda_dlls() -> None:
    """ctranslate2 needs cuBLAS/cuDNN for CUDA 12. The nvidia-*-cu12 wheels ship them in site-packages/nvidia/*/bin,
    which Windows does not search by default."""
    import importlib.util
    import os
    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations or []) if spec else []:
        for bin_dir in Path(root).glob("*/bin"):
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(bin_dir))


_register_cuda_dlls()


def cuda_available() -> bool:
    if _cuda_broken:
        return False
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def default_model() -> str:
    return "large-v3" if cuda_available() else "small"


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


# ---------- ffmpeg output parsers (pure, tested) ----------

def parse_scenes(log: str) -> list[float]:
    return [float(t) for t in re.findall(r"pts_time:([\d.]+)", log)]


def parse_silence(log: str, duration: float) -> list[dict]:
    out, start = [], None
    for kind, val in re.findall(r"silence_(start|end): (-?[\d.]+)", log):
        if kind == "start":
            start = max(0.0, float(val))
        elif start is not None:
            out.append({"start": start, "end": float(val)})
            start = None
    if start is not None:
        out.append({"start": start, "end": float(duration)})
    return out


def parse_loudness(log: str) -> dict:
    i = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    lra = re.findall(r"LRA:\s+(-?[\d.]+) LU\b", log)
    return {"integrated_lufs": float(i[-1]) if i else None, "loudness_range_lu": float(lra[-1]) if lra else None}


def keyframe_times(scenes: list[float], duration: float, minimum: int = 6, maximum: int = 12) -> list[float]:
    if duration <= 0:
        return []
    times = [min(1.0, duration * 0.05)] + [s + 0.3 for s in scenes if s + 0.3 < duration]
    for k in range(minimum):
        if len(times) >= minimum:
            break
        t = duration * (k + 0.5) / minimum
        if all(abs(t - x) > 2 for x in times):
            times.append(t)
    times.sort()
    if len(times) > maximum:
        times = [times[round(i * (len(times) - 1) / (maximum - 1))] for i in range(maximum)]
    return [round(t, 2) for t in times]


# ---------- stages ----------

def acquire_url(url: str, job_dir: Path, on_progress) -> tuple[Path, dict]:
    import yt_dlp

    def hook(d):
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        if d.get("status") == "downloading" and total:
            on_progress(min(0.99, d.get("downloaded_bytes", 0) / total))

    opts = {"outtmpl": str(job_dir / "source.%(ext)s"), "format": "bv*[height<=1080]+ba/b",
            "merge_output_format": "mp4", "quiet": True, "no_warnings": True, "noprogress": True,
            "noplaylist": True, "progress_hooks": [hook],
            # a live stream never ends and would hold the single worker forever
            "match_filter": yt_dlp.utils.match_filter_func("!is_live"), "socket_timeout": 30,
            "max_filesize": 8 * 1024**3}  # ponytail: 8 GB cap; raise it if you transcribe feature films
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info.get("_type") == "playlist" or not (info.get("requested_downloads") or info.get("filepath")):
            raise RuntimeError("nothing was downloaded (live stream, playlist page, or file over 8 GB?)")
        downloads = info.get("requested_downloads") or []
        path = Path(downloads[0]["filepath"]) if downloads else Path(ydl.prepare_filename(info))
    meta = {k: info.get(k) for k in ("title", "uploader", "channel", "upload_date", "duration", "view_count",
                                     "like_count", "webpage_url", "description", "tags", "categories")}
    if meta.get("description"):
        meta["description"] = meta["description"][:2000]
    return path, meta


def probe(src: Path) -> dict:
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)], 120)
    if p.returncode:
        raise RuntimeError(f"ffprobe could not read the file: {p.stderr.strip()[:300]}")
    data = json.loads(p.stdout)
    fmt = data.get("format", {})
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = None
    if v and v.get("avg_frame_rate", "0/0") != "0/0":
        n, d = v["avg_frame_rate"].split("/")
        fps = round(int(n) / int(d), 2) if int(d) else None
    try:
        duration = float(fmt.get("duration") or 0)
    except ValueError:  # browser-recorded webm reports "N/A"
        duration = 0.0
    return {
        "container": fmt.get("format_long_name"), "duration": duration,
        "size_mb": round(int(fmt.get("size") or 0) / 1e6, 2), "bitrate_kbps": round(int(fmt.get("bit_rate") or 0) / 1000),
        "title_tag": (fmt.get("tags") or {}).get("title"),
        "video": {"codec": v.get("codec_name"), "width": v.get("width"), "height": v.get("height"), "fps": fps,
                  "pix_fmt": v.get("pix_fmt")} if v else None,
        "audio": {"codec": a.get("codec_name"), "sample_rate": int(a.get("sample_rate") or 0),
                  "channels": a.get("channels"), "layout": a.get("channel_layout")} if a else None,
    }


def _load(name: str, device: str):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device=device, compute_type="float16" if device == "cuda" else "int8")


def get_model(name: str):
    key = (name, "cuda" if cuda_available() else "cpu")
    if key not in _model:
        _model.clear()
        _model[key] = _load(*key)
    return _model[key]


def with_cuda_fallback(work):
    """Run work() (which calls get_model). CUDA libraries are loaded lazily at first encode, so a GPU that is
    detected but unusable only fails here: retry once on CPU and stay there."""
    global _cuda_broken
    try:
        return work()
    except RuntimeError as e:
        if _cuda_broken or not re.search(r"cublas|cudnn|cuda", str(e), re.I):
            raise
        _cuda_broken = True
        _model.clear()
        return work()


def transcribe(src: Path, job_dir: Path, model_name: str, language: str | None, on_progress) -> dict:
    wav = job_dir / "audio.wav"
    p = run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    if p.returncode:
        raise RuntimeError(f"audio extraction failed: {p.stderr.strip()[:300]}")
    def work():
        segs, info = get_model(model_name).transcribe(str(wav), language=language or None, vad_filter=True, beam_size=5)
        segments = []
        for s in segs:  # generator: decoding happens here
            segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                             "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3)})
            if info.duration:
                on_progress(min(0.99, s.end / info.duration))
        return segments, info

    with _model_lock:
        segments, info = with_cuda_fallback(work)
    top = sorted(info.all_language_probs or [], key=lambda x: -x[1])[:5]
    return {"segments": segments, "language": info.language, "language_probability": round(info.language_probability, 3),
            "language_candidates": [{"code": c, "p": round(pr, 3)} for c, pr in top], "model": model_name,
            "device": "cuda" if cuda_available() else "cpu"}


def scan_audio(src: Path, duration: float) -> dict:
    p = run(["ffmpeg", "-nostats", "-i", str(src), "-vn", "-af", "silencedetect=n=-35dB:d=0.6,ebur128", "-f", "null", "-"])
    silences = parse_silence(p.stderr, duration)
    silent = sum(s["end"] - s["start"] for s in silences)
    return {**parse_loudness(p.stderr), "silences": silences[:200], "silence_seconds": round(silent, 2),
            "silence_ratio": round(silent / duration, 3) if duration else 0}


def scan_video(src: Path, job_dir: Path, duration: float, on_progress) -> dict:
    p = run(["ffmpeg", "-nostats", "-i", str(src), "-an", "-vf", "scale=320:-2,select='gt(scene,0.35)',showinfo",
             "-fps_mode", "vfr", "-f", "null", "-"])
    scenes = parse_scenes(p.stderr)
    on_progress(0.6)
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frames: list[dict] = []
    for i, t in enumerate(keyframe_times(scenes, duration)):
        out = frames_dir / f"f_{i:03}.jpg"
        r = run(["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", str(src), "-frames:v", "1",
                 "-vf", "scale='min(960,iw)':-2", "-q:v", "3", str(out)])
        if r.returncode == 0 and out.exists():
            frames.append({"index": len(frames), "time": t, "file": out.name})
    on_progress(0.99)
    shots = [b - a for a, b in zip([0.0] + scenes, scenes + [duration]) if b > a]
    return {"scene_cuts": scenes, "scene_count": len(scenes) + 1,
            "cuts_per_minute": round(len(scenes) / duration * 60, 2) if duration else 0,
            "avg_shot_seconds": round(sum(shots) / len(shots), 2) if shots else None, "keyframes": frames}


# ---------- orchestration ----------

def process(job: dict, update, src: Path | None, url: str | None, opts: dict) -> dict:
    """update(stage, status=None, progress=None, **result_fields) is the only way this touches shared job state."""
    job_dir = Path(job["dir"])
    result: dict = {"options": opts, "source": {"url": url}}

    update("acquire", "running", 0)
    if url:
        src, result["source"]["page"] = acquire_url(url, job_dir, lambda p: update("acquire", progress=p))
    assert src is not None, "either a file or a URL is required"
    result["source"]["file"] = src.name
    update("acquire", "done", 1)

    update("probe", "running", 0)
    meta = result["media"] = probe(src)
    duration = meta["duration"]
    update("probe", "done", 1, media=meta, source=result["source"])

    use_ai = ai.available()

    def guarded(stage, fn):
        """The video/audio scans are extras: a failure there must not throw away a finished transcript."""
        def run():
            try:
                return fn()
            except Exception as e:
                update(stage, "failed", 1, note=str(e)[:200])
                return None
        return run

    def t_transcribe():
        if not meta["audio"]:
            update("transcribe", "skipped", 1, note="no audio stream")
            return None
        update("transcribe", "running", 0)
        r = transcribe(src, job_dir, opts["model"], opts["language"], lambda p: update("transcribe", progress=p))
        update("transcribe", "done", 1, transcript=r)
        return r

    def t_visual():
        if not meta["video"]:
            update("visual_scan", "skipped", 1)
            return None
        update("visual_scan", "running", 0)
        v = scan_video(src, job_dir, duration, lambda p: update("visual_scan", progress=p))
        update("visual_scan", "done", 1, visual=v)
        if use_ai and opts["ocr"] and v["keyframes"]:
            update("screen_text", "running", 0.1)
            try:
                update("screen_text", "done", 1, screen_text=ai.read_frames(job_dir / "frames", v["keyframes"]))
            except Exception as e:
                update("screen_text", "failed", 1, screen_text_error=str(e))
        return v

    def t_audio():
        if not meta["audio"]:
            update("audio_scan", "skipped", 1)
            return None
        update("audio_scan", "running", 0.2)
        a = scan_audio(src, duration)
        update("audio_scan", "done", 1, audio=a)
        return a

    with ThreadPoolExecutor(3) as pool:  # the simultaneous scan: speech, picture and sound in parallel
        fv = pool.submit(guarded("visual_scan", t_visual))
        fa = pool.submit(guarded("audio_scan", t_audio))
        transcript = pool.submit(t_transcribe).result()
        fv.result(), fa.result()

    segments = transcript["segments"] if transcript else []
    if not duration and segments:  # container without a duration: the speech timeline is the best we have
        duration = segments[-1]["end"]
    update("text_analysis", "running", 0.5)
    stats = text_stats(segments, duration)
    update("text_analysis", "done", 1, stats=stats)
    (job_dir / "transcript.txt").write_text(export(segments, "txt"), encoding="utf-8")
    (job_dir / "transcript.srt").write_text(export(segments, "srt"), encoding="utf-8")

    if opts["ai"] and use_ai and segments:
        update("ai_review", "running", 0.1)
        try:
            update("ai_review", "done", 1, review=ai.review(segments, transcript, result["source"], stats))
        except Exception as e:
            update("ai_review", "failed", 1, review_error=str(e))
    elif opts["ai"] and not ai.available():
        update("ai_review", "skipped", 1, review_error="AI layer not configured: set ANTHROPIC_API_KEY and restart")

    target = (opts.get("translate_to") or "").strip()
    if target and segments:
        update("translate", "running", 0)
        try:
            if use_ai:
                tr = ai.translate(segments, target, lambda p: update("translate", progress=p))
            elif target.lower() in ("en", "english"):
                tr = whisper_to_english(src, job_dir, opts["model"])
            else:
                raise RuntimeError("translation to languages other than English needs the AI layer (ANTHROPIC_API_KEY)")
            (job_dir / "translation.txt").write_text(export(tr["segments"], "txt"), encoding="utf-8")
            update("translate", "done", 1, translation=tr)
        except Exception as e:
            update("translate", "failed", 1, translation_error=str(e))
    return result


def whisper_to_english(src: Path, job_dir: Path, model_name: str) -> dict:
    if model_name in ("large-v3-turbo", "distil-large-v3.5"):  # trained without the translate task
        model_name = "large-v3"

    def work():
        segs, _ = get_model(model_name).transcribe(str(job_dir / "audio.wav"), task="translate", vad_filter=True)
        return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]

    with _model_lock:
        segments = with_cuda_fallback(work)
    return {"target": "English", "engine": f"whisper:{model_name}", "segments": segments}


def save_upload(fileobj, filename: str, job_dir: Path) -> Path:
    suffix = Path(filename or "").suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
        suffix = ".bin"
    dest = job_dir / f"source{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(fileobj, f, 1024 * 1024)
    return dest
