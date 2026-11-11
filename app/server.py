"""FastAPI backend for the Local English TTS Readout Studio."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.engines import (
    OUTPUTS_DIR,
    REF_CLIPS_DIR,
    EngineManager,
    GenerationCancelledError,
    find_ffmpeg,
    get_engine_catalog,
    get_system_info,
    resample_audio,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class _QuietJobPollFilter(logging.Filter):
    """Suppress noisy 200 OK polling lines for /api/jobs/ in Uvicorn access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ('"GET /api/jobs/' in msg and '" 200' in msg)


logging.getLogger("uvicorn.access").addFilter(_QuietJobPollFilter())

app = FastAPI(title="OmniVox", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

engine_manager = EngineManager()
_synth_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_sessions: Dict[str, Dict[str, Any]] = {}


class GenerateRequest(BaseModel):
    engine_id: str = Field(..., description="Engine identifier from /api/status")
    text: str = Field(..., min_length=1)
    voice: str = Field(...)
    params: Dict[str, Any] = Field(default_factory=dict)
    reference_id: Optional[str] = Field(default=None)


class RegenerateChunkRequest(BaseModel):
    text: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="UI file missing")
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return {
        "system": get_system_info(),
        "engines": get_engine_catalog(),
    }


@app.post("/api/reference")
async def upload_reference(file: UploadFile = File(...)) -> Dict[str, Any]:
    ref_id = f"ref_{uuid.uuid4().hex[:10]}"
    raw_ext = Path(file.filename or "clip.wav").suffix.lower() or ".wav"
    raw_path = REF_CLIPS_DIR / f"{ref_id}_raw{raw_ext}"
    wav_path = REF_CLIPS_DIR / f"{ref_id}.wav"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    raw_path.write_bytes(data)

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(raw_path), "-ar", "24000", "-ac", "1", str(wav_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        raw_path.unlink(missing_ok=True)
        if proc.returncode != 0 or not wav_path.exists():
            raise HTTPException(status_code=400, detail=f"Could not decode audio clip: {proc.stderr.strip()}")
    else:
        if raw_ext != ".wav":
            raw_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="ffmpeg not found; please upload a WAV file")
        raw_path.rename(wav_path)

    return {
        "reference_id": ref_id,
        "filename": file.filename,
        "path": str(wav_path),
    }


@app.post("/api/generate")
def start_generation(req: GenerateRequest) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    session_id = f"readout_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ref_path = None
    if req.reference_id:
        candidate = REF_CLIPS_DIR / f"{req.reference_id}.wav"
        if not candidate.exists():
            raise HTTPException(status_code=404, detail="Reference audio clip not found")
        ref_path = str(candidate)

    cancel_event = threading.Event()
    _jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "queued",
        "cancelling": False,
        "current_chunk": 0,
        "total_chunks": 0,
        "current_text": "",
        "error": None,
        "result": None,
        "_cancel_event": cancel_event,
    }

    def _worker() -> None:
        if cancel_event.is_set():
            _jobs[job_id]["status"] = "cancelled"
            return
        _jobs[job_id]["status"] = "running"

        def _on_progress(idx: int, total: int, chunk_text: str) -> None:
            _jobs[job_id]["current_chunk"] = idx + 1
            _jobs[job_id]["total_chunks"] = total
            _jobs[job_id]["current_text"] = chunk_text

        try:
            with _synth_lock:
                if cancel_event.is_set():
                    raise GenerationCancelledError("Generation cancelled.")
                result = engine_manager.synthesize_full(
                    session_id=session_id,
                    engine_id=req.engine_id,
                    text=req.text,
                    voice=req.voice,
                    params=req.params,
                    reference_audio_path=ref_path,
                    progress_cb=_on_progress,
                    cancel_event=cancel_event,
                )
            _sessions[session_id] = result
            session_meta_path = OUTPUTS_DIR / session_id / "meta.json"
            session_meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            _jobs[job_id]["result"] = result
            _jobs[job_id]["status"] = "done"
        except GenerationCancelledError:
            _jobs[job_id]["status"] = "cancelled"
            _jobs[job_id]["cancelling"] = False
        except Exception as exc:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": job_id, "session_id": session_id}


def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _public_job(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["_cancel_event"].set()
    if job["status"] == "queued":
        job["status"] = "cancelled"
    elif job["status"] == "running":
        job["cancelling"] = True
    return _public_job(job)


def _load_session(session_id: str) -> Dict[str, Any]:
    if session_id in _sessions:
        return _sessions[session_id]
    meta_path = OUTPUTS_DIR / session_id / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _sessions[session_id] = meta
        return meta
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/sessions/{session_id}/chunks/{chunk_idx}/regenerate")
def regenerate_chunk(session_id: str, chunk_idx: int, req: RegenerateChunkRequest) -> Dict[str, Any]:
    session_meta = _load_session(session_id)
    chunks = session_meta["chunks"]
    if chunk_idx < 0 or chunk_idx >= len(chunks):
        raise HTTPException(status_code=404, detail="Chunk index out of range")

    chunk_text = (req.text if req.text is not None else chunks[chunk_idx]["text"]).strip()
    if not chunk_text:
        raise HTTPException(status_code=400, detail="Chunk text cannot be empty")
    params = req.params if req.params is not None else session_meta.get("params", {})

    with _synth_lock:
        audio, sr = engine_manager.synthesize_chunk(
            engine_id=session_meta["engine_id"],
            text=chunk_text,
            voice=session_meta["voice"],
            params=params,
            reference_audio_path=session_meta.get("reference_audio_path"),
        )
        target_sr = int(session_meta.get("sample_rate", 24000))
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)

        chunk_wav = OUTPUTS_DIR / session_id / f"chunk_{chunk_idx:03d}.wav"
        sf.write(str(chunk_wav), audio, target_sr)
        chunks[chunk_idx]["text"] = chunk_text
        session_meta["params"] = params
        updated = engine_manager.rebuild_session_audio(session_meta)
        (OUTPUTS_DIR / session_id / "meta.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")

    return updated


@app.get("/api/sessions/{session_id}/readout.wav")
def get_readout_wav(session_id: str):
    wav_path = OUTPUTS_DIR / session_id / "readout.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="WAV file not found")
    return FileResponse(str(wav_path), media_type="audio/wav", filename=f"{session_id}.wav")


@app.get("/api/sessions/{session_id}/readout.mp3")
def get_readout_mp3(session_id: str):
    mp3_path = OUTPUTS_DIR / session_id / "readout.mp3"
    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail="MP3 file not found")
    return FileResponse(
        str(mp3_path),
        media_type="audio/mpeg",
        filename=f"{session_id}.mp3",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.mp3"'},
    )


@app.get("/api/sessions/{session_id}/chunks/{chunk_idx}")
def get_chunk_wav(session_id: str, chunk_idx: int):
    chunk_path = OUTPUTS_DIR / session_id / f"chunk_{chunk_idx:03d}.wav"
    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Chunk WAV not found")
    return FileResponse(str(chunk_path), media_type="audio/wav")
