"""Monophonic transcription for vocals and bass.

A pitch tracker gives a continuous f0 contour; the note segmentation below is
what turns that contour into discrete notes, and it is the part worth testing
carefully. It is a pure function over arrays, so it needs no model.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from song2midi.audio.io import to_mono
from song2midi.midi.model import Note
from song2midi.transcription.base import sort_notes, velocity_from_energy

A4_HZ = 440.0
A4_MIDI = 69
DEFAULT_FRAME_STEP = 0.01
MIN_PYIN_FRAME_LENGTH = 2048
PYIN_PERIODS_PER_FRAME = 2

# Voicing thresholds belong to the tracker, not the stem. pyin's voiced_prob is
# the YIN trough mass on voiced pitch bins, on a scale nothing like crepe's
# periodicity: a real bass note sits at 0.24-0.49 while band-limited noise sits
# at 0.010, so 0.5 would discard everything.
PYIN_CONFIDENCE_THRESHOLD = 0.05
CREPE_CONFIDENCE_THRESHOLD = 0.5

# torchcrepe frames per forward pass. The same value on every device on purpose:
# torchcrepe.predict runs its Viterbi decoder once per batch, so the batch size
# is not a pure memory knob - dropping 512 to 128 moves individual f0 estimates
# by ~30 cents, enough to flip a borderline note by a semitone. The note cache
# is keyed by stem and transcriber, not by device, so a CPU run and a CUDA run
# have to agree. It only ever changes on out-of-memory, where the alternative
# is no notes at all.
#
# Cost, measured as peak process RSS: 512 frames = +1.45 GB, 256 = +0.78 GB, on
# top of 85 MB of weights. That makes crepe - not Demucs - the largest single
# allocation in the whole pipeline.
CREPE_BATCH_SIZE = 512
MIN_CREPE_BATCH_SIZE = 64


def pyin_frame_length(fmin: float, sr: int) -> int:
    """Frame length that fits at least two periods of `fmin`.

    pyin needs two periods of the lowest expected pitch inside a frame or its
    estimates fall apart. A 5-string bass reaches 31 Hz, where the default
    2048-sample frame holds barely one period — which is why bass came back
    almost empty before this was derived from fmin.
    """
    required = int(np.ceil(PYIN_PERIODS_PER_FRAME * sr / max(fmin, 1e-6)))
    return max(MIN_PYIN_FRAME_LENGTH, 1 << (required - 1).bit_length())


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
    max_gap_frames = int(round(max_gap / frame_step))

    segments: list[tuple[float, float, int, float]] = []
    start_index: int | None = None
    current_pitch: float | None = None
    gap_frames = 0

    def emit(end_index: int) -> None:
        _emit(
            segments, start_index, end_index, current_pitch,
            times, rms, frame_step, min_duration,
        )

    for index, pitch in enumerate(pitches):
        if np.isnan(pitch):
            if start_index is not None:
                gap_frames += 1
                if gap_frames > max_gap_frames:
                    emit(index - gap_frames + 1)
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

    # Velocity is a second pass: the reference percentile needs the whole track.
    velocities = velocity_from_energy([energy for *_, energy in segments])
    notes = [
        Note(start=start, end=end, pitch=pitch, velocity=int(velocity))
        for (start, end, pitch, _), velocity in zip(segments, velocities)
    ]
    return sort_notes(notes)


def _emit(
    segments: list[tuple[float, float, int, float]],
    start_index: int | None,
    end_index: int,
    pitch: float | None,
    times: NDArray,
    rms: NDArray,
    frame_step: float,
    min_duration: float,
) -> None:
    if pitch is None or start_index is None or end_index <= start_index:
        return
    start = float(times[start_index])
    end = float(times[min(end_index - 1, len(times) - 1)]) + frame_step
    if end - start < min_duration:
        return
    window = rms[start_index:end_index]
    energy = float(np.mean(window)) if window.size else 0.0
    segments.append((start, end, int(np.clip(pitch, 0, 127)), energy))


def _median_filter(values: NDArray, size: int) -> NDArray:
    if size <= 1:
        return values
    padded = np.pad(values, size // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    with warnings.catch_warnings():
        # An all-unvoiced window is a normal state, not a problem.
        warnings.simplefilter("ignore", RuntimeWarning)
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
        self.segmentation_kwargs = dict(segmentation_kwargs)
        self.segmentation_kwargs.setdefault(
            "confidence_threshold",
            CREPE_CONFIDENCE_THRESHOLD if backend == "crepe" else PYIN_CONFIDENCE_THRESHOLD,
        )

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
            frame_length=pyin_frame_length(self.fmin, sr),
            hop_length=hop_length,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        # librosa already NaNs f0 wherever its own voicing decision is False,
        # and notes_from_f0's isfinite(midi) term re-applies that, so gating on
        # voiced_prob can only tighten pyin's decision, never loosen it. That
        # gate is what rejects the aperiodic bleed which put a G4 on a bass
        # track. Its scale is nothing like crepe's, hence the separate default
        # in PYIN_CONFIDENCE_THRESHOLD.
        return np.nan_to_num(f0), times, np.nan_to_num(voiced_prob)

    def _track_crepe(self, mono, sr):
        import librosa
        import torch
        import torchcrepe

        from song2midi.device import is_out_of_memory, release_cuda, warn

        target_sr = 16000
        if sr != target_sr:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        hop_length = 160  # 10 ms at 16 kHz
        tensor = torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))[None]

        # Demucs has had a retry ladder from the start; crepe is the larger
        # allocation and had none, so a GPU that ran out here killed a run that
        # had already paid for separation.
        device, batch_size = self.device, CREPE_BATCH_SIZE
        while True:
            try:
                pitch, periodicity = torchcrepe.predict(
                    tensor,
                    target_sr,
                    hop_length=hop_length,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    model="full",
                    batch_size=batch_size,
                    device=device,
                    return_periodicity=True,
                )
                break
            except (RuntimeError, MemoryError) as exc:
                if device == "cpu" or not is_out_of_memory(exc):
                    raise
                release_cuda(torch)
                if batch_size > MIN_CREPE_BATCH_SIZE:
                    batch_size //= 2
                    warn(
                        f"crepe ran out of GPU memory; retrying with "
                        f"batch_size={batch_size}"
                    )
                else:
                    device, batch_size = "cpu", CREPE_BATCH_SIZE
                    warn("crepe ran out of GPU memory; pitch-tracking on the CPU")
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
