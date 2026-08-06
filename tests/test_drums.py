import numpy as np
import pytest

from song2midi.transcription.drums import (
    GM_CLOSED_HAT,
    GM_CRASH,
    GM_KICK,
    GM_SNARE,
    DrumTranscriber,
    classify_onset,
)

SR = 44100


def spectrum_at(freqs, peaks):
    """Build a magnitude spectrum with energy only at the given frequencies."""
    magnitude = np.zeros_like(freqs)
    for hz in peaks:
        magnitude[np.argmin(np.abs(freqs - hz))] = 1.0
    return magnitude


@pytest.fixture
def freqs():
    return np.linspace(0, SR / 2, 1025)


def test_low_energy_is_a_kick(freqs):
    assert classify_onset(spectrum_at(freqs, [60, 90]), freqs, decay_ratio=0.1) == GM_KICK


def test_mid_energy_is_a_snare(freqs):
    assert (
        classify_onset(spectrum_at(freqs, [200, 400, 700]), freqs, decay_ratio=0.1)
        == GM_SNARE
    )


def test_high_energy_with_fast_decay_is_a_closed_hat(freqs):
    assert (
        classify_onset(spectrum_at(freqs, [8000, 11000]), freqs, decay_ratio=0.05)
        == GM_CLOSED_HAT
    )


def test_high_energy_with_slow_decay_is_a_crash(freqs):
    assert (
        classify_onset(spectrum_at(freqs, [8000, 11000]), freqs, decay_ratio=0.6)
        == GM_CRASH
    )


def test_silence_yields_no_notes():
    audio = np.zeros((2, SR), dtype=np.float32)
    assert DrumTranscriber().transcribe(audio, SR) == []


def kick_hit(length, sr=SR, freq=60.0):
    t = np.arange(length) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-np.linspace(0, 12, length))).astype(
        np.float32
    )


def test_detects_the_right_number_of_hits():
    mono = np.zeros(SR * 2, dtype=np.float32)
    hit_times = [0.2, 0.7, 1.2, 1.7]
    length = int(0.08 * SR)
    for time in hit_times:
        start = int(time * SR)
        mono[start : start + length] += kick_hit(length)

    notes = DrumTranscriber().transcribe(np.stack([mono, mono]), SR)

    assert len(notes) == len(hit_times)
    for note, expected in zip(notes, hit_times):
        assert note.start == pytest.approx(expected, abs=0.05)


def test_kick_pattern_is_classified_as_kicks():
    mono = np.zeros(SR * 2, dtype=np.float32)
    length = int(0.08 * SR)
    for time in (0.2, 0.7, 1.2):
        start = int(time * SR)
        mono[start : start + length] += kick_hit(length, freq=55.0)

    notes = DrumTranscriber().transcribe(np.stack([mono, mono]), SR)

    assert {n.pitch for n in notes} == {GM_KICK}


def test_all_notes_have_positive_duration():
    mono = np.zeros(SR, dtype=np.float32)
    length = int(0.05 * SR)
    mono[1000 : 1000 + length] = (
        np.random.default_rng(1).standard_normal(length).astype(np.float32) * 0.3
    )

    for note in DrumTranscriber().transcribe(np.stack([mono, mono]), SR):
        assert note.end > note.start


def test_notes_are_returned_sorted():
    mono = np.zeros(SR * 2, dtype=np.float32)
    length = int(0.08 * SR)
    for time in (1.5, 0.3, 0.9):
        start = int(time * SR)
        mono[start : start + length] += kick_hit(length)

    notes = DrumTranscriber().transcribe(np.stack([mono, mono]), SR)

    assert [n.start for n in notes] == sorted(n.start for n in notes)
