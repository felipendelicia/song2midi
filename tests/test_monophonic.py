import numpy as np
import pytest

from song2midi.transcription.monophonic import (
    MonophonicTranscriber,
    hz_to_midi_float,
    notes_from_f0,
)

HOP = 0.01


def frames(count):
    return np.arange(count) * HOP


def constant_f0(midi_pitch, count):
    return np.full(count, 440.0 * 2 ** ((midi_pitch - 69) / 12))


def test_hz_to_midi_float_maps_a440_to_69():
    assert hz_to_midi_float(np.array([440.0]))[0] == pytest.approx(69.0)


def test_hz_to_midi_float_marks_non_positive_as_nan():
    assert np.isnan(hz_to_midi_float(np.array([0.0, -1.0]))).all()


def test_steady_pitch_becomes_one_note():
    count = 100
    notes = notes_from_f0(
        constant_f0(69, count), frames(count), np.ones(count), np.full(count, 0.5)
    )
    assert len(notes) == 1
    assert notes[0].pitch == 69
    assert notes[0].duration == pytest.approx(0.99, abs=0.02)


def test_pitch_change_splits_into_two_notes():
    f0 = np.concatenate([constant_f0(60, 50), constant_f0(64, 50)])
    notes = notes_from_f0(f0, frames(100), np.ones(100), np.full(100, 0.5))
    assert [n.pitch for n in notes] == [60, 64]


def test_unvoiced_frames_break_a_note():
    f0 = constant_f0(60, 100)
    confidence = np.ones(100)
    confidence[40:60] = 0.0  # 200 ms gap, well over max_gap
    notes = notes_from_f0(f0, frames(100), confidence, np.full(100, 0.5))
    assert [n.pitch for n in notes] == [60, 60]


def test_a_short_gap_does_not_break_a_note():
    f0 = constant_f0(60, 100)
    confidence = np.ones(100)
    confidence[50:52] = 0.0  # 20 ms, under max_gap
    notes = notes_from_f0(f0, frames(100), confidence, np.full(100, 0.5))
    assert len(notes) == 1


def test_vibrato_does_not_fragment_the_note():
    count = 200
    base = constant_f0(69, count)
    wobble = base * (1 + 0.02 * np.sin(np.linspace(0, 12 * np.pi, count)))
    notes = notes_from_f0(wobble, frames(count), np.ones(count), np.full(count, 0.5))
    assert len(notes) == 1
    assert notes[0].pitch == 69


def test_notes_shorter_than_the_minimum_are_dropped():
    f0 = constant_f0(60, 100)
    f0[50:52] = 440.0 * 2 ** ((72 - 69) / 12)  # 20 ms blip
    notes = notes_from_f0(f0, frames(100), np.ones(100), np.full(100, 0.5))
    assert 72 not in [n.pitch for n in notes]


def test_velocity_tracks_loudness():
    count = 100
    loud = notes_from_f0(
        constant_f0(60, count), frames(count), np.ones(count), np.full(count, 1.0)
    )
    quiet = notes_from_f0(
        constant_f0(60, count),
        frames(count),
        np.ones(count),
        np.concatenate([np.full(count - 1, 0.1), [1.0]]),
    )
    assert loud[0].velocity > quiet[0].velocity


def test_all_unvoiced_yields_no_notes():
    assert notes_from_f0(constant_f0(60, 50), frames(50), np.zeros(50), np.full(50, 0.5)) == []


def test_notes_are_returned_sorted():
    f0 = np.concatenate([constant_f0(67, 40), constant_f0(60, 40)])
    notes = notes_from_f0(f0, frames(80), np.ones(80), np.full(80, 0.5))
    assert [n.start for n in notes] == sorted(n.start for n in notes)


@pytest.mark.slow
def test_pyin_backend_finds_a_sine():
    sr = 22050
    t = np.arange(sr * 2) / sr
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    audio = np.stack([mono, mono]).astype(np.float32)

    notes = MonophonicTranscriber(fmin=60.0, fmax=400.0, backend="pyin").transcribe(audio, sr)

    assert notes
    longest = max(notes, key=lambda n: n.duration)
    assert longest.pitch == 57  # A3


@pytest.mark.slow
def test_silence_yields_no_notes():
    audio = np.zeros((2, 22050), dtype=np.float32)
    assert MonophonicTranscriber(fmin=60.0, fmax=400.0).transcribe(audio, 22050) == []
