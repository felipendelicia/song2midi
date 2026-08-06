import numpy as np
import pytest

from song2midi.transcription.polyphonic import PolyphonicTranscriber

SR = 44100


def tone(midi_pitch: int, seconds: float, sr: int = SR) -> np.ndarray:
    """A sawtooth at the given pitch.

    Sawtooths carry the harmonic content Basic Pitch was trained on; a pure
    sine is unusually hard for it.
    """
    freq = 440.0 * 2 ** ((midi_pitch - 69) / 12)
    t = np.arange(int(seconds * sr)) / sr
    wave = np.zeros_like(t)
    for harmonic in range(1, 8):
        wave += np.sin(2 * np.pi * freq * harmonic * t) / harmonic
    envelope = np.minimum(1.0, np.linspace(0, 20, t.size)) * np.linspace(1.0, 0.3, t.size)
    return (wave / np.abs(wave).max() * envelope * 0.8).astype(np.float32)


@pytest.mark.slow
def test_detects_a_single_sustained_pitch():
    audio = np.stack([tone(69, 1.5)] * 2)

    notes = PolyphonicTranscriber().transcribe(audio, SR)

    assert notes, "expected at least one note"
    longest = max(notes, key=lambda n: n.duration)
    assert longest.pitch == 69
    assert longest.start == pytest.approx(0.0, abs=0.15)


@pytest.mark.slow
def test_returns_notes_sorted_by_start():
    silence = np.zeros(int(0.3 * SR), dtype=np.float32)
    mono = np.concatenate([tone(60, 0.8), silence, tone(67, 0.8)])
    audio = np.stack([mono] * 2)

    notes = PolyphonicTranscriber().transcribe(audio, SR)

    assert [n.start for n in notes] == sorted(n.start for n in notes)
    assert {60, 67} <= {n.pitch for n in notes}


@pytest.mark.slow
def test_silence_yields_no_notes():
    audio = np.zeros((2, SR), dtype=np.float32)
    assert PolyphonicTranscriber().transcribe(audio, SR) == []
