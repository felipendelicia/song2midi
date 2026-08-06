"""Audio decoding.

Validates eagerly so a bad input fails at second 0 rather than after three
minutes of separation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from song2midi.errors import UnsupportedAudioError

TARGET_SR = 44100

# libsndfile 1.2.2 — the copy the soundfile wheels bundle — links libmpg123 and
# libopus, so mp3 and opus need no external decoder. That matters most on
# Windows, where ffmpeg is absent from a stock machine and pip cannot install
# it: routing mp3 through here is what makes a standalone .exe useful at all.
NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".opus"}
# Containers genuinely outside libsndfile. These need ffmpeg or nothing.
FFMPEG_SUFFIXES = {".m4a", ".aac", ".wma"}
# Where a libsndfile failure is worth a second attempt through ffmpeg: a distro
# libsndfile may be built without libmpg123. An uncompressed format libsndfile
# rejects is corrupt, not exotic, so it gets no retry.
FFMPEG_FALLBACK_SUFFIXES = {".mp3", ".opus", ".ogg"}
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
        try:
            audio, sr = _load_soundfile(path)
        except UnsupportedAudioError:
            if suffix not in FFMPEG_FALLBACK_SUFFIXES or ffmpeg_executable() is None:
                raise
            audio, sr = _load_ffmpeg(path, target_sr)
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


def ffmpeg_executable() -> str | None:
    """Absolute path to an ffmpeg we can launch, or None.

    Returns the path rather than a boolean because `shutil.which` and the
    process launcher do not resolve a bare name the same way: on Windows
    `which` honours every PATHEXT entry while CreateProcess only appends
    `.exe`, so a boolean guard can pass while the launch fails.
    """
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for directory in _bundled_search_dirs():
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg")


def _bundled_search_dirs() -> list[Path]:
    """Where a frozen build looks for a co-located ffmpeg, in order.

    `sys._MEIPASS` is where PyInstaller's --add-binary lands; the executable's
    own directory is where a user would drop an ffmpeg next to song2midi.exe.
    Neither is on PATH, so neither is reachable via `shutil.which`.
    """
    if not getattr(sys, "frozen", False):
        return []
    directories = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        directories.append(Path(bundle))
    directories.append(Path(sys.executable).parent)
    return directories


def _load_soundfile(path: Path) -> tuple[NDArray, int]:
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # libsndfile surfaces several unrelated types
        raise UnsupportedAudioError(f"Could not decode {path}: {exc}") from exc
    return data.T, sr


def _load_ffmpeg(path: Path, target_sr: int) -> tuple[NDArray, int]:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise UnsupportedAudioError(
            f"Decoding {path.suffix} requires ffmpeg, which was not found on PATH "
            f"or alongside the executable."
        )
    command = [
        ffmpeg, "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "2", "-ar", str(target_sr), "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", "replace").strip()
        raise UnsupportedAudioError(f"Could not decode {path}: {message}") from exc
    except OSError as exc:  # not executable, wrong architecture, permissions
        raise UnsupportedAudioError(
            f"Could not decode {path}: could not run ffmpeg ({ffmpeg}): {exc}"
        ) from exc
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
