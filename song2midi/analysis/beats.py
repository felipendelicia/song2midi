"""Beat and downbeat detection.

The output feeds quantisation, which needs the real beat positions rather than
a single BPM. When only librosa is available the downbeats come back empty and
bar-level quantisation is unavailable — that degradation is intentional.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

from song2midi.audio.io import to_mono
from song2midi.midi.model import TempoMap


def detect(audio: NDArray[np.float32], sr: int, device: str = "cpu") -> TempoMap:
    try:
        return _detect_beat_this(audio, sr, device)
    except Exception as exc:  # missing package, checkpoint download failure, ...
        _warn(f"beat_this unavailable ({exc}); falling back to librosa beat tracking")
        return detect_librosa(audio, sr)


def detect_librosa(audio: NDArray[np.float32], sr: int) -> TempoMap:
    import librosa

    mono = to_mono(audio)
    if not np.any(mono):
        return TempoMap.constant()
    _, beats = librosa.beat.beat_track(y=mono, sr=sr, units="time")
    return TempoMap.from_beats(np.asarray(beats, dtype=np.float64))


def _detect_beat_this(audio: NDArray[np.float32], sr: int, device: str) -> TempoMap:
    from beat_this.inference import Audio2Beats

    mono = to_mono(audio)
    if not np.any(mono):
        return TempoMap.constant()
    beats, downbeats = Audio2Beats(device=device)(mono, sr)
    return TempoMap.from_beats(
        np.asarray(beats, dtype=np.float64),
        np.asarray(downbeats, dtype=np.float64),
    )


def _warn(message: str) -> None:
    print(f"song2midi: {message}", file=sys.stderr)
