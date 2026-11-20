"""HTTP API + UI. Run: python -m app.server   (then open http://127.0.0.1:8765)"""
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import mimetypes

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import ai, pipeline
from .analysis import export

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("VVTR_OUTPUT", ROOT.parent / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
WORKER = ThreadPoolExecutor(1)  # ponytail: one job at a time (one GPU); a queue forms naturally
ID_RE = re.compile(r"[0-9a-f]{12}")
EXPORT_TYPES = {"txt": "text/plain", "srt": "application/x-subrip", "vtt": "text/vtt", "md": "text/markdown",
                "json": "application/json"}

LOCAL_HOSTS = ["127.0.0.1", "localhost", "testserver"]  # testserver: FastAPI TestClient

app = FastAPI(title="Video Voice-to-Text Reader")
# DNS rebinding: a foreign hostname resolving to 127.0.0.1 must not reach the API.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)


@app.middleware("http")
async def same_origin_writes(request: Request, call_next):
    """CSRF: any web page can POST a form to localhost. Writes must come from this app's own page."""
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD") and origin and urlparse(origin).hostname not in LOCAL_HOSTS:
        return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def job_dir(job_id: str) -> Path:
    if not ID_RE.fullmatch(job_id):
        raise HTTPException(404, "no such job")
    return OUT / job_id


def snapshot(job: dict) -> dict:
    with LOCK:
        return json.loads(json.dumps(job))


def persist(job: dict) -> None:
    (Path(job["dir"]) / "job.json").write_text(json.dumps(snapshot(job), ensure_ascii=False, indent=1), encoding="utf-8")


def load(job_id: str) -> dict:
    d = job_dir(job_id)
    with LOCK:
        if job_id in JOBS:
            return JOBS[job_id]
    f = d / "job.json"
    if not f.exists():
        raise HTTPException(404, "no such job")
    return json.loads(f.read_text(encoding="utf-8"))


def run_job(job: dict, src: Path | None, url: str | None, opts: dict) -> None:
    def update(stage, status=None, progress=None, **fields):
        with LOCK:
            st = job["stages"].setdefault(stage, {"status": "pending", "progress": 0})
            if status:
                st["status"] = status
            if progress is not None:
                st["progress"] = round(progress, 3)
            if "note" in fields:
                st["note"] = fields.pop("note")
            job["result"].update(fields)

    with LOCK:
        job["status"] = "running"
    try:
        result = pipeline.process(job, update, src, url, opts)
        with LOCK:
            job["result"].update(result)
            job["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        with LOCK:
            job["status"], job["error"] = "failed", str(e)
            for st in job["stages"].values():
                if st["status"] == "running":
                    st["status"] = "failed"
    finally:
        job["finished"] = time.time()
        (Path(job["dir"]) / "audio.wav").unlink(missing_ok=True)
        persist(job)


def mark_interrupted() -> None:
    """A job that was queued/running when the server stopped will never finish: say so instead of hiding it."""
    for f in OUT.glob("*/job.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            if j.get("status") in ("queued", "running") and j.get("id") not in JOBS:
                j["status"], j["error"] = "interrupted", "the server stopped before this job finished"
                f.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
        except (OSError, ValueError):
            pass


mark_interrupted()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
def config():
    return {"ai": ai.available(), "ai_model": ai.MODEL, "cuda": pipeline.cuda_available(),
            "models": pipeline.MODELS, "default_model": pipeline.default_model()}


@app.post("/api/jobs")
def create_job(file: UploadFile | None = File(None), url: str = Form(""), model: str = Form(""),
               language: str = Form(""), translate_to: str = Form(""), use_ai: bool = Form(True),
               ocr: bool = Form(True)):
    url = url.strip()
    if bool(url) == bool(file and file.filename):
        raise HTTPException(400, "give either a file or a URL")
    if url and urlparse(url).scheme not in ("http", "https"):
        raise HTTPException(400, "URL must start with http:// or https://")
    if model and model not in pipeline.MODELS:
        raise HTTPException(400, f"unknown model; choose one of {pipeline.MODELS}")
    if language and not re.fullmatch(r"[a-z]{2,3}", language):
        raise HTTPException(400, "language must be an ISO code such as en, es, de")

    job_id = uuid.uuid4().hex[:12]
    d = OUT / job_id
    d.mkdir()
    src = pipeline.save_upload(file.file, file.filename, d) if not url else None
    opts = {"model": model or pipeline.default_model(), "language": language or None,
            "translate_to": translate_to.strip()[:40], "ai": use_ai, "ocr": ocr}
    job = {"id": job_id, "dir": str(d), "created": time.time(), "status": "queued", "error": None,
           "name": url or file.filename, "stages": {}, "result": {}}
    with LOCK:
        JOBS[job_id] = job
    persist(job)  # visible in History (and deletable) even if the server stops mid-job
    WORKER.submit(run_job, job, src, url or None, opts)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    seen, items = set(), []
    with LOCK:
        live = [dict(id=j["id"], name=j["name"], created=j["created"], status=j["status"]) for j in JOBS.values()]
    for j in live:
        seen.add(j["id"])
        items.append(j)
    for f in OUT.glob("*/job.json"):
        if f.parent.name not in seen:
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
                items.append(dict(id=j["id"], name=j["name"], created=j["created"], status=j["status"]))
            except (OSError, ValueError, KeyError):
                pass
    return sorted(items, key=lambda j: -j["created"])[:100]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = load(job_id)
    out = snapshot(job) if job_id in JOBS else job
    out.pop("dir", None)
    return out


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = load(job_id)
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "job is still running")
    with LOCK:
        JOBS.pop(job_id, None)
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/media")
def media(job_id: str):
    src = load(job_id)["result"].get("source", {}).get("file")
    if not src:
        raise HTTPException(404, "media not ready")
    path = job_dir(job_id) / Path(src).name
    mt = mimetypes.guess_type(path.name)[0] or ""
    # only ever serve the upload as media, never as something a browser would render as a page
    return FileResponse(path, media_type=mt if mt.startswith(("video/", "audio/")) else "application/octet-stream")


@app.get("/api/jobs/{job_id}/frames/{name}")
def frame(job_id: str, name: str):
    if not re.fullmatch(r"f_\d{3}\.jpg", name):
        raise HTTPException(404)
    path = job_dir(job_id) / "frames" / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


class Seg(BaseModel):
    start: float
    end: float
    text: str


class ExportReq(BaseModel):
    fmt: str
    kind: str = "transcript"
    segments: list[Seg]
    save: bool = False


@app.post("/api/jobs/{job_id}/export")
def export_job(job_id: str, req: ExportReq):
    if req.fmt not in EXPORT_TYPES or req.kind not in ("transcript", "translation"):
        raise HTTPException(400, "bad format or kind")
    job = load(job_id)
    segs = [s.model_dump() for s in req.segments]
    title = job["result"].get("source", {}).get("page", {}) or {}
    body = export(segs, req.fmt, title=title.get("title") or job["name"])
    if req.save:  # edits survive a reload: stored beside the raw output, raw is never overwritten
        (job_dir(job_id) / f"{req.kind}_edited.{req.fmt}").write_text(body, encoding="utf-8")
        with LOCK:
            job["result"][f"{req.kind}_edited"] = segs
        persist(job)
    return Response(body, media_type=f"{EXPORT_TYPES[req.fmt]}; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{req.kind}-{job_id}.{req.fmt}"'})


def main():
    import uvicorn
    import webbrowser
    port = int(os.environ.get("PORT", 8765))
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
