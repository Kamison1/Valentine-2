"""Loading and exporting audio.

Internally all audio is a float32 array of shape ``(2, n_samples)`` at 44.1 kHz.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import soundfile as sf

from . import AudioLoadError, DJMixError

SAMPLE_RATE = 44100
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac")
MIN_DURATION_SEC = 10.0


def check_supported(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Unsupported file type '{ext or '(none)'}' for {os.path.basename(path)}. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."
        )


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load ``path`` as stereo float32 at ``sr``. Mono files are duplicated."""
    import librosa

    if not os.path.isfile(path):
        raise AudioLoadError(f"File not found: {path}")
    check_supported(path)
    try:
        with warnings.catch_warnings():
            # librosa warns when it falls back to audioread for mp3; that's fine.
            warnings.simplefilter("ignore")
            y, _ = librosa.load(path, sr=sr, mono=False)
    except Exception as exc:  # decoding errors come in many flavours
        hint = ""
        if path.lower().endswith(".mp3") and shutil.which("ffmpeg") is None:
            hint = " MP3 decoding may need ffmpeg - install it and make sure it's on PATH."
        raise AudioLoadError(
            f"Could not decode {os.path.basename(path)}: {exc}.{hint}"
        ) from exc

    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] > 2:
        # Down-mix surround to stereo by keeping the front pair.
        y = y[:2]
    elif y.shape[0] == 1:
        y = np.vstack([y, y])
    y = np.ascontiguousarray(y, dtype=np.float32)

    duration = y.shape[1] / sr
    if duration < MIN_DURATION_SEC:
        raise AudioLoadError(
            f"{os.path.basename(path)} is only {duration:.1f}s long; "
            f"need at least {MIN_DURATION_SEC:.0f}s of audio to mix."
        )
    if not np.any(np.abs(y) > 1e-4):
        raise AudioLoadError(f"{os.path.basename(path)} appears to be silent.")
    return y


def to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=0) if y.ndim == 2 else y


def min_max_envelope(mono: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bucket min and max of a mono signal, for drawing waveforms without aliasing."""
    edges = np.linspace(0, len(mono), min(buckets, len(mono)) + 1).astype(int)
    return np.minimum.reduceat(mono, edges[:-1]), np.maximum.reduceat(mono, edges[:-1])


def prevent_clipping(y: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y


def export_wav(path: str, y: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, y.T, sr, subtype="PCM_16")
    return path


def export_mp3(path: str, y: np.ndarray, sr: int = SAMPLE_RATE, bitrate: str = "320k") -> str:
    """Encode to MP3 with ffmpeg (libmp3lame)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DJMixError("ffmpeg is not installed, so MP3 export is unavailable (WAV was still written).")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, y.T, sr, subtype="PCM_16")
        proc = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", tmp_path,
             "-codec:a", "libmp3lame", "-b:a", bitrate, path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise DJMixError(f"ffmpeg failed to write {path}: {proc.stderr.strip()}")
    finally:
        os.unlink(tmp_path)
    return path
