"""Audio decoding.

Validates eagerly so a bad input fails at second 0 rather than after three
minutes of separation.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from song2midi.errors import UnsupportedAudioError

TARGET_SR = 44100

# Formats libsndfile handles directly; everything else goes through ffmpeg.
NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
FFMPEG_SUFFIXES = {".mp3", ".m4a", ".aac", ".opus", ".wma"}
SUPPORTED_SUFFIXES = NATIVE_SUFFIXES | FFMPEG_SUFFIXES


def load(path: Path, target_sr: int = TARGET_SR) -> tuple[NDArray[np.float32], int]:
    """Decode `path` to a `(2, N)` float32 array at `target_sr`."""
    path = Path(path)
    if not path.is_file():
        raise UnsupportedAudioError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedAudioError(
            f"Unsupported audio format {suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix in NATIVE_SUFFIXES:
        audio, sr = _load_soundfile(path)
    else:
        audio, sr = _load_ffmpeg(path, target_sr)

    audio = _to_stereo(audio)
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def to_mono(audio: NDArray) -> NDArray[np.float32]:
    """Average channels of a `(channels, N)` array. 1-D input passes through."""
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=0).astype(np.float32)


def _load_soundfile(path: Path) -> tuple[NDArray, int]:
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # libsndfile surfaces several unrelated types
        raise UnsupportedAudioError(f"Could not decode {path}: {exc}") from exc
    return data.T, sr


def _load_ffmpeg(path: Path, target_sr: int) -> tuple[NDArray, int]:
    if shutil.which("ffmpeg") is None:
        raise UnsupportedAudioError(
            f"Decoding {path.suffix} requires ffmpeg, which is not on PATH."
        )
    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "2", "-ar", str(target_sr), "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", "replace").strip()
        raise UnsupportedAudioError(f"Could not decode {path}: {message}") from exc
    flat = np.frombuffer(result.stdout, dtype=np.float32)
    if flat.size == 0:
        raise UnsupportedAudioError(f"Could not decode {path}: ffmpeg produced no audio")
    return flat.reshape(-1, 2).T.copy(), target_sr


def _to_stereo(audio: NDArray) -> NDArray:
    if audio.ndim == 1:
        return np.stack([audio, audio])
    if audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    if audio.shape[0] > 2:
        return audio[:2]
    return audio


def _resample(audio: NDArray, sr: int, target_sr: int) -> NDArray:
    import librosa  # imported lazily: ~2 s of import time we skip when rates match

    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=-1)
