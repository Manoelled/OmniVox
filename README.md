# OmniVox

Made by Manoelle Diokno.

OmniVox is a local, token-free English text-to-speech studio that unifies multiple open-source neural and ONNX TTS engines inside a single browser interface. Paste a script, pick an English voice, tune dramaticness and pacing, play/pause/skip/regenerate by sentence, and export MP3 or WAV audio on your own hardware.

## Features

- **No API Keys or Paid Tokens:** Runs locally on CPU or NVIDIA CUDA GPU (with optional zero-GPU online Edge TTS).
- **Live System and Voice Cache Status:** Shows which engines and English voices are already cached on disk alongside estimated download sizes before you generate.
- **Cancel Active Generation:** Stop queued or multi-sentence synthesis cleanly mid-run.
- **Engine-Specific Tuning and Dramaticness:**
  - **Kokoro 82M (`hexgrad/kokoro`):** Fast English synthesis (~1s per minute of audio on an RTX 3060 Ti), speaking speed, secondary voice blending, and pause pacing.
  - **Chatterbox (`resemble-ai/chatterbox`):** Native Dramaticness (`exaggeration`) slider, CFG pacing control, vocal variety (`temperature`), and zero-shot voice cloning.
  - **Chatterbox Turbo (`resemble-ai/chatterbox`):** Streamlined 350M English model with native paralinguistic tags (`[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[cough]`).
  - **Piper ONNX (`OHF-Voice/piper1-gpl`):** 100% offline CPU synthesis with speed and expressiveness (`noise_scale`) sliders.
  - **KittenTTS 0.8 & 2 (`KittenML/KittenTTS`):** Lightweight 80M ONNX CPU voices plus KittenTTS 2 character voices, emotion/event tags (`[joyful]`, `<laugh>`), and voice cloning.
  - **Edge TTS (`travisvn/openai-edge-tts`):** Token-free online English voices with speed and pitch controls and zero GPU usage.
- **Interactive Playback and Export:** Sentence-level chunking for Play/Pause, `-10s` / `+10s`, Prev/Next Line skipping, single-line regeneration, auto-play toggle, and MP3/WAV export via `ffmpeg`.

## Project Structure

```text
OmniVox/
├── app/
│   ├── __init__.py
│   ├── engines.py          # Engine adapters, cache/size inspection, chunking, MP3 conversion
│   ├── server.py           # FastAPI backend, job polling, cancellation, and audio endpoints
│   └── static/
│       └── index.html      # Monochrome frosted-glass Web UI
├── requirements.txt
├── start_studio.bat        # Windows one-click launcher
├── start_studio.ps1        # PowerShell launcher (http://127.0.0.1:7860)
└── README.md
```

## Setup

### Prerequisites

- Python 3.10 to 3.12
- `ffmpeg` installed and available on `PATH` (or at `C:\ffmpeg\bin\ffmpeg.exe` on Windows) for MP3 export and voice cloning clip conversion
- Optional: NVIDIA GPU with CUDA-enabled PyTorch (`torch==2.6.0` + `torchaudio==2.6.0`)

### Installation

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install PyTorch (CUDA 12.4 example for NVIDIA GPUs):

   ```powershell
   pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ```

3. Install OmniVox and TTS engine dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Launch OmniVox:

   ```powershell
   .\start_studio.ps1
   ```

   Or double-click `start_studio.bat`, then open **http://127.0.0.1:7860** in your browser.

## Notes on Model Downloads

Model weights are downloaded lazily on first use and cached locally:
- **Lightweight:** Piper (~60 MB to 115 MB per voice), KittenTTS 0.8 (~78 MB), Kokoro 82M (~314 MB), Edge TTS (0 MB local).
- **Large:** Chatterbox (~3.0 GB), Chatterbox Turbo (~1.5 GB), KittenTTS 2 (~1.2 GB).

## Upstream Repositories

- [hexgrad/kokoro](https://github.com/hexgrad/kokoro)
- [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
- [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)
- [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS)
- [travisvn/openai-edge-tts](https://github.com/travisvn/openai-edge-tts)
