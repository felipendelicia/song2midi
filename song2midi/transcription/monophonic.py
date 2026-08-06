"""Monophonic transcription for vocals and bass.

A pitch tracker gives a continuous f0 contour; the note segmentation below is
what turns that contour into discrete notes, and it is the part worth testing
carefully. It is a pure function over arrays, so it needs no model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from song2midi.audio.io import to_mono
from song2midi.midi.model import Note
from song2midi.transcription.base import sort_notes

A4_HZ = 440.0
A4_MIDI = 69
DEFAULT_FRAME_STEP = 0.01


def hz_to_midi_float(f0_hz: NDArray) -> NDArray:
    """Convert Hz to fractional MIDI pitch. Non-positive input becomes NaN."""
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = A4_MIDI + 12.0 * np.log2(f0_hz / A4_HZ)
    return np.where(f0_hz > 0, midi, np.nan)


def notes_from_f0(
    f0_hz: NDArray,
    times: NDArray,
    confidence: NDArray,
    rms: NDArray,
    *,
    confidence_threshold: float = 0.5,
    min_duration: float = 0.05,
    max_gap: float = 0.06,
    median_filter_frames: int = 5,
) -> list[Note]:
    """Segment an f0 contour into notes.

    Four steps: drop low-confidence frames, convert to fractional MIDI, median
    filter to kill vibrato jitter, then group runs of equal rounded pitch,
    bridging unvoiced gaps shorter than `max_gap`.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)

    midi = hz_to_midi_float(f0_hz)
    voiced = (confidence >= confidence_threshold) & np.isfinite(midi)
    if not voiced.any():
        return []

    smoothed = _median_filter(midi, median_filter_frames)
    pitches = np.where(voiced, np.round(smoothed), np.nan)

    frame_step = (
        float(np.median(np.diff(times))) if times.size > 1 else DEFAULT_FRAME_STEP
    )
    peak_rms = float(np.max(rms)) if rms.size and np.max(rms) > 0 else 1.0
    max_gap_frames = int(round(max_gap / frame_step))

    notes: list[Note] = []
    start_index: int | None = None
    current_pitch: float | None = None
    gap_frames = 0

    def emit(end_index: int) -> None:
        _emit(
            notes, start_index, end_index, current_pitch,
            times, rms, peak_rms, frame_step, min_duration,
        )

    for index, pitch in enumerate(pitches):
        if np.isnan(pitch):
            if start_index is not None:
                gap_frames += 1
                if gap_frames > max_gap_frames:
                    emit(index - gap_frames)
                    start_index, current_pitch, gap_frames = None, None, 0
            continue

        if start_index is None:
            start_index, current_pitch, gap_frames = index, pitch, 0
        elif pitch != current_pitch:
            emit(index - gap_frames)
            start_index, current_pitch, gap_frames = index, pitch, 0
        else:
            gap_frames = 0

    if start_index is not None:
        emit(len(pitches) - gap_frames)

    return sort_notes(notes)


def _emit(
    notes: list[Note],
    start_index: int | None,
    end_index: int,
    pitch: float | None,
    times: NDArray,
    rms: NDArray,
    peak_rms: float,
    frame_step: float,
    min_duration: float,
) -> None:
    if pitch is None or start_index is None or end_index <= start_index:
        return
    start = float(times[start_index])
    end = float(times[min(end_index, len(times) - 1)]) + frame_step
    if end - start < min_duration:
        return
    segment = rms[start_index:end_index]
    loudness = float(np.mean(segment)) / peak_rms if segment.size else 0.5
    notes.append(
        Note(
            start=start,
            end=end,
            pitch=int(np.clip(pitch, 0, 127)),
            velocity=int(np.clip(round(loudness * 127), 1, 127)),
        )
    )


def _median_filter(values: NDArray, size: int) -> NDArray:
    if size <= 1:
        return values
    padded = np.pad(values, size // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(windows, axis=-1)


class MonophonicTranscriber:
    """Pitch-tracks a single-voice stem.

    `pyin` needs no model and is accurate in the bass register, which is why
    bass uses it. `crepe` is stronger on vocals, where the register is wider and
    the timbre less periodic.
    """

    def __init__(
        self,
        fmin: float,
        fmax: float,
        backend: str = "pyin",
        device: str = "cpu",
        **segmentation_kwargs,
    ) -> None:
        self.fmin = fmin
        self.fmax = fmax
        self.backend = backend
        self.device = device
        self.segmentation_kwargs = segmentation_kwargs

    def transcribe(self, audio: NDArray[np.float32], sr: int) -> list[Note]:
        mono = to_mono(audio)
        if not np.any(mono):
            return []
        if self.backend == "crepe":
            f0, times, confidence = self._track_crepe(mono, sr)
        else:
            f0, times, confidence = self._track_pyin(mono, sr)
        rms = self._frame_rms(mono, sr, times)
        return notes_from_f0(f0, times, confidence, rms, **self.segmentation_kwargs)

    def _track_pyin(self, mono, sr):
        import librosa

        hop_length = 256
        f0, _, voiced_prob = librosa.pyin(
            y=mono,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        return np.nan_to_num(f0), times, np.nan_to_num(voiced_prob)

    def _track_crepe(self, mono, sr):
        import librosa
        import torch
        import torchcrepe

        target_sr = 16000
        if sr != target_sr:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        hop_length = 160  # 10 ms at 16 kHz
        tensor = torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))[None]
        pitch, periodicity = torchcrepe.predict(
            tensor,
            target_sr,
            hop_length=hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            model="full",
            batch_size=512,
            device=self.device,
            return_periodicity=True,
        )
        f0 = pitch[0].cpu().numpy()
        confidence = periodicity[0].cpu().numpy()
        times = np.arange(len(f0)) * hop_length / target_sr
        return f0, times, confidence

    @staticmethod
    def _frame_rms(mono, sr, times):
        import librosa

        if times.size > 1:
            hop_length = max(1, int(round(float(np.median(np.diff(times))) * sr)))
        else:
            hop_length = 256
        rms = librosa.feature.rms(
            y=mono, frame_length=hop_length * 4, hop_length=hop_length
        )[0]
        if len(rms) < len(times):
            rms = np.pad(rms, (0, len(times) - len(rms)), mode="edge")
        return rms[: len(times)]
