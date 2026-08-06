"""Drum transcription by onset detection and spectral band classification.

This is a heuristic, not a trained model, and it is the weakest stage in the
pipeline: it confuses toms with kicks and does not distinguish open from closed
hi-hats. That is a deliberate trade-off — there is no maintained,
pip-installable drum transcriber worth the dependency, and an approximate
kick/snare/hat pattern is a usable starting point for editing in a DAW. The
`Transcriber` protocol makes replacing this a local change if it ever stops
being good enough.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from song2midi.audio.io import to_mono
from song2midi.midi.model import Note
from song2midi.transcription.base import sort_notes

GM_KICK = 36
GM_SNARE = 38
GM_CLOSED_HAT = 42
GM_CRASH = 49

LOW_BAND = (20.0, 150.0)
MID_BAND = (150.0, 800.0)
HIGH_BAND = (6000.0, 16000.0)

NOTE_DURATION = 0.1
ATTACK_WINDOW = 0.05
DECAY_WINDOW = (0.15, 0.30)
CRASH_DECAY_THRESHOLD = 0.3
N_FFT = 2048
MIN_ATTACK_SAMPLES = 16


def band_energy(spectrum: NDArray, freqs: NDArray, band: tuple[float, float]) -> float:
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(spectrum[mask] ** 2))


def classify_onset(spectrum: NDArray, freqs: NDArray, decay_ratio: float) -> int:
    """Map one onset's spectrum to a General MIDI drum note."""
    low = band_energy(spectrum, freqs, LOW_BAND)
    mid = band_energy(spectrum, freqs, MID_BAND)
    high = band_energy(spectrum, freqs, HIGH_BAND)

    if high > low and high > mid:
        return GM_CRASH if decay_ratio > CRASH_DECAY_THRESHOLD else GM_CLOSED_HAT
    if low > mid:
        return GM_KICK
    return GM_SNARE


class DrumTranscriber:
    def __init__(self, onset_delta: float = 0.07) -> None:
        self.onset_delta = onset_delta

    def transcribe(self, audio: NDArray[np.float32], sr: int) -> list[Note]:
        import librosa

        mono = to_mono(audio)
        if not np.any(mono):
            return []

        onset_envelope = librosa.onset.onset_strength(y=mono, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sr,
            units="time",
            backtrack=True,
            delta=self.onset_delta,
        )
        if len(onsets) == 0:
            return []

        peak = float(np.max(np.abs(mono))) or 1.0
        freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)

        notes = []
        for onset in onsets:
            attack = _window(mono, sr, onset, onset + ATTACK_WINDOW)
            if attack.size < MIN_ATTACK_SAMPLES:
                continue
            spectrum = np.abs(np.fft.rfft(attack * np.hanning(attack.size), n=N_FFT))
            tail = _window(mono, sr, onset + DECAY_WINDOW[0], onset + DECAY_WINDOW[1])
            attack_rms = _rms(attack)
            decay_ratio = _rms(tail) / (attack_rms or 1.0)

            notes.append(
                Note(
                    start=float(onset),
                    end=float(onset) + NOTE_DURATION,
                    pitch=classify_onset(spectrum, freqs, decay_ratio),
                    velocity=int(np.clip(round(attack_rms / peak * 127 * 2), 1, 127)),
                )
            )
        return sort_notes(notes)


def _window(mono: NDArray, sr: int, start: float, end: float) -> NDArray:
    return mono[max(0, int(start * sr)) : min(len(mono), int(end * sr))]


def _rms(samples: NDArray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
