"""Unified engine adapters, chunking, VRAM management, and MP3 export for Local TTS Studio."""

from __future__ import annotations

import gc
import importlib.util
import io
import os
import re
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


class GenerationCancelledError(Exception):
    """Raised when a user cancels an active synthesis job."""

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PIPER_DIR = MODELS_DIR / "piper"
REF_CLIPS_DIR = MODELS_DIR / "references"
OUTPUTS_DIR = BASE_DIR / "outputs"

for _d in (MODELS_DIR, PIPER_DIR, REF_CLIPS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg executable on PATH or default Windows install path."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    win_default = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
    if win_default.exists():
        return str(win_default)
    return None


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_system_info() -> Dict[str, Any]:
    """Return GPU, CUDA, and ffmpeg status for the UI header."""
    gpu_name = None
    vram_total_gb = None
    vram_free_gb = None
    cuda_ok = False
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_free_gb = round(free_b / (1024**3), 2)
            vram_total_gb = round(total_b / (1024**3), 2)
    except Exception:
        pass

    return {
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "vram_free_gb": vram_free_gb,
        "vram_total_gb": vram_total_gb,
        "ffmpeg_available": find_ffmpeg() is not None,
    }


def split_text_chunks(text: str, max_chars: int = 260) -> List[str]:
    """Split text into sentence-sized chunks suitable for neural TTS and sentence skipping."""
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n+", cleaned) if p.strip()]
    chunks: List[str] = []

    for para in paragraphs:
        # Split on sentence boundaries while keeping punctuation
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[(<])|(?<=[.!?])\s+$", para)
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                # Sub-split overly long sentences on commas/semicolons
                parts = re.split(r"(?<=[,;:])\s+", sent)
                sub = ""
                for part in parts:
                    if sub and len(sub) + 1 + len(part) > max_chars:
                        chunks.append(sub)
                        sub = part
                    else:
                        sub = f"{sub} {part}".strip() if sub else part
                if sub:
                    chunks.append(sub)
            elif current and len(current) + 1 + len(sent) > max_chars:
                chunks.append(current)
                current = sent
            else:
                current = f"{current} {sent}".strip() if current else sent
        if current:
            chunks.append(current)

    return chunks


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float32 audio using linear interpolation when sample rates differ."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / float(orig_sr)
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def normalize_peak(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    """Safely normalize float32 audio without clipping."""
    if len(audio) == 0:
        return audio.astype(np.float32)
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    max_val = float(np.max(np.abs(arr)))
    if max_val > 1e-6:
        arr = arr * (peak / max_val)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> Path:
    """Encode a WAV file to MP3 using the local ffmpeg binary."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or add it to PATH for MP3 export.")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(mp3_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not mp3_path.exists():
        raise RuntimeError(f"ffmpeg MP3 conversion failed: {proc.stderr.strip()}")
    return mp3_path


# ---------------------------------------------------------------------------
# Engine Catalog & Dynamic UI Schemas (English-focused, token-free)
# ---------------------------------------------------------------------------

KOKORO_VOICES = [
    {"id": "af_heart", "label": "Heart (US Female - Warm, Natural)"},
    {"id": "af_bella", "label": "Bella (US Female - Expressive)"},
    {"id": "af_nicole", "label": "Nicole (US Female - Soft / ASMR)"},
    {"id": "af_sarah", "label": "Sarah (US Female - Crisp Narrator)"},
    {"id": "af_sky", "label": "Sky (US Female - Bright)"},
    {"id": "af_nova", "label": "Nova (US Female - Clear)"},
    {"id": "af_alloy", "label": "Alloy (US Female - Balanced)"},
    {"id": "af_jessica", "label": "Jessica (US Female - Conversational)"},
    {"id": "am_michael", "label": "Michael (US Male - Deep Narrator)"},
    {"id": "am_adam", "label": "Adam (US Male - Classic)"},
    {"id": "am_fenrir", "label": "Fenrir (US Male - Bold)"},
    {"id": "am_eric", "label": "Eric (US Male - Friendly)"},
    {"id": "am_liam", "label": "Liam (US Male - Young)"},
    {"id": "am_onyx", "label": "Onyx (US Male - Rich)"},
    {"id": "am_puck", "label": "Puck (US Male - Lively)"},
    {"id": "bf_emma", "label": "Emma (UK Female - Refined)"},
    {"id": "bf_isabella", "label": "Isabella (UK Female - Warm)"},
    {"id": "bf_alice", "label": "Alice (UK Female - Clear)"},
    {"id": "bf_lily", "label": "Lily (UK Female - Gentle)"},
    {"id": "bm_george", "label": "George (UK Male - Classic)"},
    {"id": "bm_lewis", "label": "Lewis (UK Male - Storytelling)"},
    {"id": "bm_daniel", "label": "Daniel (UK Male - Broadcaster)"},
    {"id": "bm_fable", "label": "Fable (UK Male - Theatrical)"},
]

PIPER_VOICES = [
    {"id": "en_US-lessac-medium", "label": "Lessac (US - Balanced Natural)"},
    {"id": "en_US-amy-medium", "label": "Amy (US Female - Clear)"},
    {"id": "en_US-ryan-high", "label": "Ryan (US Male - High Quality)"},
    {"id": "en_US-libritts_r-medium", "label": "LibriTTS (US - Audiobook)"},
    {"id": "en_US-joe-medium", "label": "Joe (US Male - Conversational)"},
    {"id": "en_GB-alba-medium", "label": "Alba (UK/Scottish Female)"},
    {"id": "en_GB-cori-high", "label": "Cori (UK Female - High Quality)"},
    {"id": "en_GB-northern_english_male-medium", "label": "Northern Male (UK)"},
    {"id": "en_GB-alan-medium", "label": "Alan (UK Male - Crisp)"},
]

KITTEN_ONNX_VOICES = [
    {"id": "Bella", "label": "Bella (Female)"},
    {"id": "Jasper", "label": "Jasper (Male)"},
    {"id": "Luna", "label": "Luna (Female)"},
    {"id": "Bruno", "label": "Bruno (Male)"},
    {"id": "Rosie", "label": "Rosie (Female)"},
    {"id": "Hugo", "label": "Hugo (Male)"},
    {"id": "Kiki", "label": "Kiki (Female)"},
    {"id": "Leo", "label": "Leo (Male)"},
]

KITTEN2_VOICES = [
    {"id": "Laurence", "label": "Laurence (Dramatic Stage Actor, Male)"},
    {"id": "Victoria", "label": "Victoria (Regal Narrator, Female)"},
    {"id": "Martha", "label": "Martha (Weathered Radio Journalist, Female)"},
    {"id": "Herbert", "label": "Herbert (Wise Elderly Professor, Male)"},
    {"id": "Maeve", "label": "Maeve (Warm Storyteller, Female)"},
    {"id": "Marcus", "label": "Marcus (Smooth Radio DJ, Male)"},
    {"id": "Bella", "label": "Bella (Clear Female)"},
    {"id": "Jasper", "label": "Jasper (Clear Male)"},
    {"id": "Luna", "label": "Luna (Soft Female)"},
    {"id": "Bruno", "label": "Bruno (Grounded Male)"},
    {"id": "Kiki", "label": "Kiki (Bright Female)"},
    {"id": "Leo", "label": "Leo (Energetic Male)"},
    {"id": "Victor", "label": "Victor (Controlled Fury, Male)"},
    {"id": "Raven", "label": "Raven (Intense Young, Female)"},
    {"id": "Sable", "label": "Sable (Hushed Conspirator, Female)"},
    {"id": "Walter", "label": "Walter (Warm Grandfather, Male)"},
    {"id": "Eleanor", "label": "Eleanor (Somber Historian, Female)"},
    {"id": "Dante", "label": "Dante (Playful Smooth, Male)"},
    {"id": "Claire", "label": "Claire (Playful Professional, Female)"},
    {"id": "clone_uploaded", "label": "Use Uploaded Reference Clip (Voice Clone)"},
]

EDGE_VOICES = [
    {"id": "en-US-AriaNeural", "label": "Aria (US Female - Expressive)"},
    {"id": "en-US-GuyNeural", "label": "Guy (US Male - Newscast/Warm)"},
    {"id": "en-US-JennyNeural", "label": "Jenny (US Female - Natural)"},
    {"id": "en-US-ChristopherNeural", "label": "Christopher (US Male - Reliable)"},
    {"id": "en-US-AvaMultilingualNeural", "label": "Ava (US Female - Rich)"},
    {"id": "en-US-AndrewMultilingualNeural", "label": "Andrew (US Male - Smooth)"},
    {"id": "en-GB-SoniaNeural", "label": "Sonia (UK Female - Clear)"},
    {"id": "en-GB-RyanNeural", "label": "Ryan (UK Male - Crisp)"},
    {"id": "en-AU-NatashaNeural", "label": "Natasha (AU Female)"},
    {"id": "en-AU-WilliamNeural", "label": "William (AU Male)"},
]


def _hf_hub_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cache_dir(repo_id: str) -> Path:
    return _hf_hub_dir() / f"models--{repo_id.replace('/', '--')}"


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return round(total / (1024 * 1024), 1)


def _format_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    if mb >= 10:
        return f"{mb:.0f} MB"
    if mb > 0:
        return f"{mb:.1f} MB"
    return "0 MB"


def _hf_find_cached_file(repo_id: str, filename: str) -> Optional[Path]:
    snapshots = _hf_repo_cache_dir(repo_id) / "snapshots"
    if not snapshots.exists():
        return None
    for snap in snapshots.iterdir():
        if snap.is_dir():
            candidate = snap / filename
            if candidate.exists():
                return candidate
    return None


def _decorate_kokoro_voices(voices: List[Dict[str, str]], model_installed: bool) -> List[Dict[str, Any]]:
    out = []
    for v in voices:
        vid = v["id"]
        cached = _hf_find_cached_file("hexgrad/Kokoro-82M", f"voices/{vid}.pt")
        if cached is not None:
            mb = round(cached.stat().st_size / (1024 * 1024), 2)
            out.append({**v, "installed": True, "size_mb": mb, "size_label": f"Installed · {mb:.1f} MB"})
        else:
            est = 0.5 if model_installed else 314.0
            lbl = "Needs ~0.5 MB" if model_installed else "Needs ~0.5 MB (+313 MB model)"
            out.append({**v, "installed": False, "size_mb": est, "size_label": lbl})
    return out


def _decorate_piper_voices(voices: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for v in voices:
        vid = v["id"]
        onnx_path = PIPER_DIR / f"{vid}.onnx"
        if onnx_path.exists():
            mb = round(onnx_path.stat().st_size / (1024 * 1024), 1)
            out.append({**v, "installed": True, "size_mb": mb, "size_label": f"Installed · {mb:.0f} MB"})
        else:
            est = 115.0 if vid.endswith("-high") else 60.0
            out.append({**v, "installed": False, "size_mb": est, "size_label": f"Needs ~{est:.0f} MB"})
    return out


def _decorate_bundled_voices(
    voices: List[Dict[str, str]],
    model_installed: bool,
    model_mb: float,
    est_model_mb: float,
) -> List[Dict[str, Any]]:
    out = []
    for v in voices:
        if v["id"] == "clone_uploaded":
            lbl = "Uses your clip" if model_installed else f"Uses clip (+~{_format_mb(est_model_mb)} model)"
            out.append({**v, "installed": model_installed, "size_mb": 0.0, "size_label": lbl})
        elif model_installed:
            out.append({**v, "installed": True, "size_mb": model_mb, "size_label": "Installed (in model)"})
        else:
            out.append(
                {
                    **v,
                    "installed": False,
                    "size_mb": est_model_mb,
                    "size_label": f"Needs ~{_format_mb(est_model_mb)} model",
                }
            )
    return out


def get_engine_catalog() -> List[Dict[str, Any]]:
    """Return engine metadata, cache installation status, size previews, voices, and tuning controls."""
    kokoro_mb = _dir_size_mb(_hf_repo_cache_dir("hexgrad/Kokoro-82M"))
    kokoro_installed = _hf_find_cached_file("hexgrad/Kokoro-82M", "kokoro-v1_0.pth") is not None

    cb_mb = _dir_size_mb(_hf_repo_cache_dir("ResembleAI/chatterbox"))
    cb_installed = _hf_find_cached_file("ResembleAI/chatterbox", "t3_cfg.safetensors") is not None

    cbt_mb = _dir_size_mb(_hf_repo_cache_dir("ResembleAI/chatterbox-turbo"))
    cbt_installed = cbt_mb > 100.0

    piper_voices_decorated = _decorate_piper_voices(PIPER_VOICES)
    piper_installed_count = sum(1 for v in piper_voices_decorated if v["installed"])
    piper_mb = _dir_size_mb(PIPER_DIR)

    kitten_onnx_mb = _dir_size_mb(_hf_repo_cache_dir("KittenML/kitten-tts-mini-0.8"))
    kitten_onnx_installed = _hf_find_cached_file("KittenML/kitten-tts-mini-0.8", "kitten_tts_mini_v0_8.onnx") is not None

    kitten2_mb = _dir_size_mb(_hf_repo_cache_dir("KittenML/kitten-tts-2"))
    kitten2_installed = kitten2_mb > 100.0

    return [
        {
            "id": "kokoro",
            "name": "Kokoro 82M",
            "repo": "hexgrad/kokoro",
            "badge": "Offline GPU/CPU · ~1s / min",
            "description": "Fastest high-quality English model on GPU or CPU. Natural US/UK English voices with voice blending.",
            "available": _has_module("kokoro"),
            "installed": kokoro_installed,
            "installed_size_mb": kokoro_mb,
            "estimated_download_mb": 314,
            "size_summary": f"Installed · {_format_mb(kokoro_mb)}" if kokoro_installed else "Download needed · ~314 MB",
            "supports_cloning": False,
            "tags": [],
            "voices": _decorate_kokoro_voices(KOKORO_VOICES, kokoro_installed),
            "controls": [
                {
                    "key": "speed",
                    "label": "Speaking Speed",
                    "type": "range",
                    "min": 0.6,
                    "max": 1.8,
                    "step": 0.05,
                    "default": 1.0,
                    "hint": "1.0x is natural cadence",
                },
                {
                    "key": "dramaticness",
                    "label": "Dramatic Pacing (Pause & Breath)",
                    "type": "range",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.35,
                    "hint": "Adds dramatic pauses between sentences & breath dynamics (for native neural emotion, try Chatterbox or KittenTTS 2)",
                },
                {
                    "key": "blend_voice",
                    "label": "Blend Secondary Voice",
                    "type": "select",
                    "default": "none",
                    "options": [{"id": "none", "label": "None (Pure Primary Voice)"}] + KOKORO_VOICES,
                    "hint": "Mix two English voice timbres together",
                },
                {
                    "key": "blend_ratio",
                    "label": "Secondary Voice Mix",
                    "type": "range",
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "default": 0.25,
                    "hint": "Weight of the secondary voice when blending",
                },
            ],
        },
        {
            "id": "chatterbox",
            "name": "Chatterbox (Dramatic + Cloning)",
            "repo": "resemble-ai/chatterbox",
            "badge": "Offline GPU · Native Dramaticness",
            "description": "500M neural TTS with native Dramaticness (exaggeration) slider, pacing control, and zero-shot voice cloning.",
            "available": _has_module("chatterbox"),
            "installed": cb_installed,
            "installed_size_mb": cb_mb,
            "estimated_download_mb": 3040,
            "size_summary": f"Installed · {_format_mb(cb_mb)}" if cb_installed else "Download needed · ~3.0 GB",
            "supports_cloning": True,
            "tags": [],
            "voices": _decorate_bundled_voices(
                [
                    {"id": "default_en", "label": "Default English Voice"},
                    {"id": "clone_uploaded", "label": "Use Uploaded Reference Clip (Voice Clone)"},
                ],
                cb_installed,
                cb_mb,
                3040,
            ),
            "controls": [
                {
                    "key": "dramaticness",
                    "label": "Dramaticness (Exaggeration)",
                    "type": "range",
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "default": 0.55,
                    "hint": "0.25 = calm/neutral · 0.5 = balanced · 0.9–1.5 = theatrical/intense",
                },
                {
                    "key": "cfg_weight",
                    "label": "Pacing & Adherence (CFG)",
                    "type": "range",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.5,
                    "hint": "Lower (~0.3) = slower, more expressive phrasing; Higher = tighter adherence",
                },
                {
                    "key": "temperature",
                    "label": "Vocal Variety (Temperature)",
                    "type": "range",
                    "min": 0.2,
                    "max": 1.2,
                    "step": 0.05,
                    "default": 0.8,
                    "hint": "Higher adds prosody variety; lower is steadier",
                },
            ],
        },
        {
            "id": "chatterbox_turbo",
            "name": "Chatterbox Turbo (350M + Tags)",
            "repo": "resemble-ai/chatterbox",
            "badge": "Offline GPU · Fast + [laugh]/[sigh]",
            "description": "Streamlined 350M English model with native paralinguistic tags ([laugh], [chuckle], [sigh], [gasp]).",
            "available": _has_module("chatterbox"),
            "installed": cbt_installed,
            "installed_size_mb": cbt_mb,
            "estimated_download_mb": 1500,
            "size_summary": f"Installed · {_format_mb(cbt_mb)}" if cbt_installed else "Download needed · ~1.5 GB",
            "supports_cloning": True,
            "tags": ["[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]"],
            "voices": _decorate_bundled_voices(
                [
                    {"id": "default_en", "label": "Default English Voice"},
                    {"id": "clone_uploaded", "label": "Use Uploaded Reference Clip (Voice Clone)"},
                ],
                cbt_installed,
                cbt_mb,
                1500,
            ),
            "controls": [
                {
                    "key": "temperature",
                    "label": "Vocal Expressiveness (Temperature)",
                    "type": "range",
                    "min": 0.2,
                    "max": 1.2,
                    "step": 0.05,
                    "default": 0.8,
                    "hint": "Controls delivery variation across takes",
                },
                {
                    "key": "top_p",
                    "label": "Nucleus Sampling (Top-P)",
                    "type": "range",
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.95,
                    "hint": "Lower for steadier phrasing, higher for livelier delivery",
                },
            ],
        },
        {
            "id": "piper",
            "name": "Piper ONNX",
            "repo": "OHF-Voice/piper1-gpl",
            "badge": "100% Offline CPU · Instant",
            "description": "Ultra-fast local ONNX English voices. Zero GPU required; downloads selected voice (~60–115 MB) on first run.",
            "available": _has_module("piper"),
            "installed": piper_installed_count > 0,
            "installed_size_mb": piper_mb,
            "estimated_download_mb": 60,
            "size_summary": (
                f"Installed ({piper_installed_count}/{len(piper_voices_decorated)} voices · {_format_mb(piper_mb)})"
                if piper_installed_count > 0
                else "Download needed · ~60 MB/voice"
            ),
            "supports_cloning": False,
            "tags": [],
            "voices": piper_voices_decorated,
            "controls": [
                {
                    "key": "speed",
                    "label": "Speaking Speed",
                    "type": "range",
                    "min": 0.6,
                    "max": 1.8,
                    "step": 0.05,
                    "default": 1.0,
                    "hint": "Controls Piper length_scale",
                },
                {
                    "key": "dramaticness",
                    "label": "Expressiveness (Noise Variation)",
                    "type": "range",
                    "min": 0.0,
                    "max": 1.4,
                    "step": 0.05,
                    "default": 0.67,
                    "hint": "Adjusts Piper noise_scale & rhythm variance (0.67 is standard)",
                },
            ],
        },
        {
            "id": "kitten_onnx",
            "name": "KittenTTS 0.8 Mini (ONNX)",
            "repo": "KittenML/KittenTTS",
            "badge": "Offline CPU/ONNX · 80M",
            "description": "Ultra-lightweight 80M ONNX neural English model. Runs fast without touching GPU VRAM.",
            "available": _has_module("kittenml"),
            "installed": kitten_onnx_installed,
            "installed_size_mb": kitten_onnx_mb,
            "estimated_download_mb": 78,
            "size_summary": f"Installed · {_format_mb(kitten_onnx_mb)}" if kitten_onnx_installed else "Download needed · ~78 MB",
            "supports_cloning": False,
            "tags": [],
            "voices": _decorate_bundled_voices(KITTEN_ONNX_VOICES, kitten_onnx_installed, kitten_onnx_mb, 78),
            "controls": [
                {
                    "key": "speed",
                    "label": "Speaking Speed",
                    "type": "range",
                    "min": 0.6,
                    "max": 1.8,
                    "step": 0.05,
                    "default": 1.0,
                    "hint": "1.0x is natural cadence",
                },
            ],
        },
        {
            "id": "kitten_2",
            "name": "KittenTTS 2 (Characters & Emotion)",
            "repo": "KittenML/KittenTTS",
            "badge": "Offline GPU/CPU · 492 MB emb4",
            "description": "Character voices (stage actor, radio DJ, narrator), emotion tags ([joyful], [stern]), and voice cloning.",
            "available": _has_module("kittenml"),
            "installed": kitten2_installed,
            "installed_size_mb": kitten2_mb,
            "estimated_download_mb": 1200,
            "size_summary": f"Installed · {_format_mb(kitten2_mb)}" if kitten2_installed else "Download needed · ~1.2 GB",
            "supports_cloning": True,
            "tags": [
                "[joyful]",
                "[excited]",
                "[contemplative]",
                "[stern]",
                "[sad]",
                "[tender]",
                "<laugh>",
                "<sigh>",
                "<gasp>",
                "<pause>",
            ],
            "voices": _decorate_bundled_voices(KITTEN2_VOICES, kitten2_installed, kitten2_mb, 1200),
            "controls": [
                {
                    "key": "dramaticness",
                    "label": "Dramaticness (Preset & Temperature)",
                    "type": "range",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": 0.4,
                    "hint": ">= 0.5 switches to KittenTTS 'expressive' preset; < 0.5 uses 'stable'",
                },
            ],
        },
        {
            "id": "edge_tts",
            "name": "Edge TTS (No API Token)",
            "repo": "travisvn/openai-edge-tts",
            "badge": "Online · 0 MB Local",
            "description": "Microsoft Edge neural English voices without any API token or GPU usage (requires internet connection).",
            "available": _has_module("edge_tts"),
            "installed": True,
            "installed_size_mb": 0.0,
            "estimated_download_mb": 0,
            "size_summary": "Ready · 0 MB (Streams online)",
            "supports_cloning": False,
            "tags": [],
            "voices": [
                {**v, "installed": True, "size_mb": 0.0, "size_label": "0 MB · Online"} for v in EDGE_VOICES
            ],
            "controls": [
                {
                    "key": "speed",
                    "label": "Speaking Speed",
                    "type": "range",
                    "min": 0.6,
                    "max": 1.8,
                    "step": 0.05,
                    "default": 1.0,
                    "hint": "Maps to Edge rate (%)",
                },
                {
                    "key": "pitch_hz",
                    "label": "Voice Pitch (Hz)",
                    "type": "range",
                    "min": -25,
                    "max": 25,
                    "step": 1,
                    "default": 0,
                    "hint": "Shifts pitch deeper (-) or brighter (+)",
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Lazy Model Cache + 8 GB VRAM Guard
# ---------------------------------------------------------------------------

class EngineManager:
    """Manages lazy loading, 8 GB VRAM swapping, and chunked synthesis across engines."""

    def __init__(self) -> None:
        self._kokoro_model = None
        self._kokoro_pipelines: Dict[str, Any] = {}
        self._chatterbox_model = None
        self._chatterbox_turbo_model = None
        self._piper_voices: Dict[str, Any] = {}
        self._kitten_onnx_model = None
        self._kitten2_model = None
        self._active_gpu_family: Optional[str] = None

    def _ensure_gpu_room(self, target_family: str) -> None:
        """Unload other heavy PyTorch GPU models so 8 GB VRAM stays comfortable."""
        if self._active_gpu_family == target_family:
            return

        if target_family != "kokoro" and self._kokoro_model is not None:
            self._kokoro_pipelines.clear()
            self._kokoro_model = None
        if target_family != "chatterbox" and self._chatterbox_model is not None:
            self._chatterbox_model = None
        if target_family != "chatterbox_turbo" and self._chatterbox_turbo_model is not None:
            self._chatterbox_turbo_model = None
        if target_family != "kitten_2" and self._kitten2_model is not None:
            self._kitten2_model = None

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self._active_gpu_family = target_family

    # -- Kokoro --
    def _get_kokoro_pipeline(self, lang_code: str):
        self._ensure_gpu_room("kokoro")
        from kokoro import KModel, KPipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._kokoro_model is None:
            self._kokoro_model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        if lang_code not in self._kokoro_pipelines:
            self._kokoro_pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",
                model=self._kokoro_model,
                device=device,
            )
        return self._kokoro_pipelines[lang_code]

    # -- Chatterbox --
    def _get_chatterbox(self):
        self._ensure_gpu_room("chatterbox")
        if self._chatterbox_model is None:
            import torch
            from chatterbox.tts import ChatterboxTTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
        return self._chatterbox_model

    def _get_chatterbox_turbo(self):
        self._ensure_gpu_room("chatterbox_turbo")
        if self._chatterbox_turbo_model is None:
            import torch
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._chatterbox_turbo_model = ChatterboxTurboTTS.from_pretrained(device=device)
        return self._chatterbox_turbo_model

    # -- Piper --
    def _get_piper_voice(self, voice_id: str):
        if voice_id in self._piper_voices:
            return self._piper_voices[voice_id]
        from piper import PiperVoice
        from piper.download_voices import download_voice

        onnx_path = PIPER_DIR / f"{voice_id}.onnx"
        json_path = PIPER_DIR / f"{voice_id}.onnx.json"
        if not onnx_path.exists() or not json_path.exists():
            download_voice(voice_id, PIPER_DIR)
        voice = PiperVoice.load(str(onnx_path))
        self._piper_voices[voice_id] = voice
        return voice

    # -- KittenTTS --
    def _get_kitten_onnx(self):
        if self._kitten_onnx_model is None:
            from kittenml import KittenTTS

            self._kitten_onnx_model = KittenTTS("KittenML/kitten-tts-mini-0.8")
        return self._kitten_onnx_model

    def _get_kitten2(self):
        self._ensure_gpu_room("kitten_2")
        if self._kitten2_model is None:
            import torch
            from kittenml import KittenTTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._kitten2_model = KittenTTS("KittenML/kitten-tts-2", weights="emb4", device=device)
        return self._kitten2_model

    # -- Synthesis per chunk --
    def synthesize_chunk(
        self,
        engine_id: str,
        text: str,
        voice: str,
        params: Dict[str, Any],
        reference_audio_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize one text chunk and return (float32_audio, sample_rate)."""
        if engine_id == "kokoro":
            import torch

            lang_code = "b" if voice.startswith("b") else "a"
            pipe = self._get_kokoro_pipeline(lang_code)
            speed = float(params.get("speed", 1.0))
            dramaticness = float(params.get("dramaticness", 0.35))
            blend_voice = params.get("blend_voice", "none")
            blend_ratio = float(params.get("blend_ratio", 0.25))

            voice_arg: Any = voice
            if blend_voice and blend_voice != "none" and blend_voice != voice and blend_ratio > 0:
                p1 = pipe.load_single_voice(voice)
                p2_pipe = self._get_kokoro_pipeline("b" if blend_voice.startswith("b") else "a")
                p2 = p2_pipe.load_single_voice(blend_voice)
                voice_arg = ((1.0 - blend_ratio) * p1 + blend_ratio * p2).to(dtype=torch.float32)

            parts = []
            for _, _, audio in pipe(text, voice=voice_arg, speed=speed, split_pattern=None):
                if audio is not None and len(audio) > 0:
                    arr = audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
                    parts.append(arr.astype(np.float32))
            merged = np.concatenate(parts) if parts else np.zeros(2400, dtype=np.float32)

            # Apply dramatic pause tail if dramaticness > 0.35
            if dramaticness > 0.35 and text.rstrip().endswith(("!", "?", "...", "—")):
                extra_sil = int(24000 * (dramaticness - 0.35) * 0.45)
                merged = np.concatenate([merged, np.zeros(extra_sil, dtype=np.float32)])
            return normalize_peak(merged), 24000

        if engine_id == "chatterbox":
            model = self._get_chatterbox()
            exaggeration = float(params.get("dramaticness", 0.55))
            cfg_weight = float(params.get("cfg_weight", 0.5))
            temperature = float(params.get("temperature", 0.8))
            ref = reference_audio_path if (voice == "clone_uploaded" and reference_audio_path) else None
            wav_tensor = model.generate(
                text,
                audio_prompt_path=ref,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            arr = wav_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return normalize_peak(arr), int(model.sr)

        if engine_id == "chatterbox_turbo":
            model = self._get_chatterbox_turbo()
            temperature = float(params.get("temperature", 0.8))
            top_p = float(params.get("top_p", 0.95))
            ref = reference_audio_path if (voice == "clone_uploaded" and reference_audio_path) else None
            kwargs: Dict[str, Any] = {"temperature": temperature, "top_p": top_p}
            if ref:
                kwargs["audio_prompt_path"] = ref
            wav_tensor = model.generate(text, **kwargs)
            arr = wav_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return normalize_peak(arr), int(model.sr)

        if engine_id == "piper":
            from piper.config import SynthesisConfig

            piper_voice = self._get_piper_voice(voice)
            speed = max(0.3, float(params.get("speed", 1.0)))
            dramaticness = float(params.get("dramaticness", 0.67))
            syn_cfg = SynthesisConfig(
                length_scale=1.0 / speed,
                noise_scale=dramaticness,
                noise_w_scale=max(0.1, min(1.5, dramaticness * 1.15)),
                normalize_audio=True,
            )
            parts = []
            sr = piper_voice.config.sample_rate
            for chunk in piper_voice.synthesize(text, syn_config=syn_cfg):
                parts.append(chunk.audio_float_array.astype(np.float32))
                sr = chunk.sample_rate
            merged = np.concatenate(parts) if parts else np.zeros(int(sr * 0.1), dtype=np.float32)
            return normalize_peak(merged), int(sr)

        if engine_id == "kitten_onnx":
            model = self._get_kitten_onnx()
            speed = float(params.get("speed", 1.0))
            audio = model.generate(text, voice=voice, speed=speed)
            return normalize_peak(np.asarray(audio, dtype=np.float32)), 24000

        if engine_id == "kitten_2":
            model = self._get_kitten2()
            dramaticness = float(params.get("dramaticness", 0.4))
            preset = "expressive" if dramaticness >= 0.5 else "stable"
            temp = round(0.55 + dramaticness * 0.45, 2)
            if voice == "clone_uploaded":
                if not reference_audio_path:
                    raise ValueError("Please upload a 5-30s reference audio clip to use voice cloning.")
                audio = model.generate(
                    text,
                    reference=reference_audio_path,
                    preset=preset,
                    temperature=temp,
                )
            else:
                audio = model.generate(
                    text,
                    voice=voice,
                    preset=preset,
                    temperature=temp,
                )
            return normalize_peak(np.asarray(audio, dtype=np.float32)), int(model.sample_rate)

        if engine_id == "edge_tts":
            import asyncio
            import edge_tts

            speed = float(params.get("speed", 1.0))
            pitch_hz = int(round(float(params.get("pitch_hz", 0))))
            rate_pct = int(round((speed - 1.0) * 100))
            rate_str = f"{rate_pct:+d}%"
            pitch_str = f"{pitch_hz:+d}Hz"

            tmp_mp3 = OUTPUTS_DIR / f"_edge_tmp_{time.time_ns()}.mp3"

            async def _run():
                comm = edge_tts.Communicate(text, voice=voice, rate=rate_str, pitch=pitch_str)
                await comm.save(str(tmp_mp3))

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Execute inside a fresh thread loop when called from async context
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(lambda: asyncio.run(_run())).result()
                else:
                    loop.run_until_complete(_run())
            except RuntimeError:
                asyncio.run(_run())

            ffmpeg = find_ffmpeg()
            if not ffmpeg:
                raise RuntimeError("ffmpeg is required to decode Edge TTS MP3 chunks for unified playback.")
            tmp_wav = tmp_mp3.with_suffix(".wav")
            subprocess.run(
                [ffmpeg, "-y", "-i", str(tmp_mp3), "-ar", "24000", "-ac", "1", str(tmp_wav)],
                capture_output=True,
                check=True,
            )
            audio, sr = sf.read(str(tmp_wav), dtype="float32")
            tmp_mp3.unlink(missing_ok=True)
            tmp_wav.unlink(missing_ok=True)
            return normalize_peak(audio), int(sr)

        raise ValueError(f"Unknown engine: {engine_id}")

    def synthesize_full(
        self,
        session_id: str,
        engine_id: str,
        text: str,
        voice: str,
        params: Dict[str, Any],
        reference_audio_path: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Synthesize full readout chunk-by-chunk, stitch WAV + MP3, and track chunk timestamps."""
        chunks_text = split_text_chunks(text)
        if not chunks_text:
            raise ValueError("Please enter some English text to synthesize.")

        t0 = time.perf_counter()
        target_sr = 24000
        dramaticness = float(params.get("dramaticness", 0.4))
        pause_sec = 0.16 + min(0.28, dramaticness * 0.18)
        pause_samples = np.zeros(int(target_sr * pause_sec), dtype=np.float32)

        session_dir = OUTPUTS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        audio_segments: List[np.ndarray] = []
        chunk_meta: List[Dict[str, Any]] = []
        cursor_sec = 0.0

        for idx, chunk_text in enumerate(chunks_text):
            if cancel_event and cancel_event.is_set():
                shutil.rmtree(session_dir, ignore_errors=True)
                raise GenerationCancelledError("Generation cancelled.")
            if progress_cb:
                progress_cb(idx, len(chunks_text), chunk_text)
            audio, sr = self.synthesize_chunk(
                engine_id=engine_id,
                text=chunk_text,
                voice=voice,
                params=params,
                reference_audio_path=reference_audio_path,
            )
            if cancel_event and cancel_event.is_set():
                shutil.rmtree(session_dir, ignore_errors=True)
                raise GenerationCancelledError("Generation cancelled.")
            if sr != target_sr:
                audio = resample_audio(audio, sr, target_sr)

            chunk_dur = len(audio) / float(target_sr)
            chunk_wav_path = session_dir / f"chunk_{idx:03d}.wav"
            sf.write(str(chunk_wav_path), audio, target_sr)

            chunk_meta.append(
                {
                    "index": idx,
                    "text": chunk_text,
                    "start_sec": round(cursor_sec, 3),
                    "end_sec": round(cursor_sec + chunk_dur, 3),
                    "duration_sec": round(chunk_dur, 3),
                    "audio_url": f"/api/sessions/{session_id}/chunks/{idx}",
                }
            )
            audio_segments.append(audio)
            cursor_sec += chunk_dur

            if idx < len(chunks_text) - 1:
                audio_segments.append(pause_samples)
                cursor_sec += pause_sec

        combined = np.concatenate(audio_segments) if audio_segments else np.zeros(2400, dtype=np.float32)
        wav_path = session_dir / "readout.wav"
        sf.write(str(wav_path), combined, target_sr)

        mp3_path = session_dir / "readout.mp3"
        mp3_ready = False
        if find_ffmpeg():
            convert_wav_to_mp3(wav_path, mp3_path)
            mp3_ready = True

        gen_sec = time.perf_counter() - t0
        audio_dur = len(combined) / float(target_sr)
        rtf = gen_sec / max(0.01, audio_dur)

        return {
            "session_id": session_id,
            "engine_id": engine_id,
            "voice": voice,
            "params": params,
            "reference_audio_path": reference_audio_path,
            "sample_rate": target_sr,
            "audio_duration_sec": round(audio_dur, 2),
            "generation_time_sec": round(gen_sec, 2),
            "realtime_factor": round(rtf, 3),
            "speedup_x": round(audio_dur / max(0.01, gen_sec), 1),
            "wav_url": f"/api/sessions/{session_id}/readout.wav",
            "mp3_url": f"/api/sessions/{session_id}/readout.mp3" if mp3_ready else None,
            "chunks": chunk_meta,
        }

    def rebuild_session_audio(self, session_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Re-stitch chunk WAV files into readout.wav + readout.mp3 after regenerating one chunk."""
        session_id = session_meta["session_id"]
        session_dir = OUTPUTS_DIR / session_id
        target_sr = int(session_meta.get("sample_rate", 24000))
        dramaticness = float(session_meta.get("params", {}).get("dramaticness", 0.4))
        pause_sec = 0.16 + min(0.28, dramaticness * 0.18)
        pause_samples = np.zeros(int(target_sr * pause_sec), dtype=np.float32)

        audio_segments: List[np.ndarray] = []
        cursor_sec = 0.0
        chunks = session_meta["chunks"]

        for idx, ch in enumerate(chunks):
            chunk_wav_path = session_dir / f"chunk_{idx:03d}.wav"
            audio, sr = sf.read(str(chunk_wav_path), dtype="float32")
            if sr != target_sr:
                audio = resample_audio(audio, sr, target_sr)
            dur = len(audio) / float(target_sr)
            ch["start_sec"] = round(cursor_sec, 3)
            ch["end_sec"] = round(cursor_sec + dur, 3)
            ch["duration_sec"] = round(dur, 3)
            audio_segments.append(audio)
            cursor_sec += dur
            if idx < len(chunks) - 1:
                audio_segments.append(pause_samples)
                cursor_sec += pause_sec

        combined = np.concatenate(audio_segments)
        wav_path = session_dir / "readout.wav"
        sf.write(str(wav_path), combined, target_sr)
        mp3_path = session_dir / "readout.mp3"
        if find_ffmpeg():
            convert_wav_to_mp3(wav_path, mp3_path)
            session_meta["mp3_url"] = f"/api/sessions/{session_id}/readout.mp3"
        session_meta["audio_duration_sec"] = round(len(combined) / float(target_sr), 2)
        return session_meta
