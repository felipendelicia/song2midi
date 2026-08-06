import numpy as np
import pytest

from song2midi.midi.model import Note, TempoMap, Track


def test_note_duration():
    assert Note(start=1.0, end=2.5, pitch=60, velocity=100).duration == 1.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 1.0, "end": 1.0, "pitch": 60, "velocity": 100},
        {"start": 2.0, "end": 1.0, "pitch": 60, "velocity": 100},
        {"start": 0.0, "end": 1.0, "pitch": 128, "velocity": 100},
        {"start": 0.0, "end": 1.0, "pitch": -1, "velocity": 100},
        {"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 0},
        {"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 128},
    ],
)
def test_note_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Note(**kwargs)


def test_tempo_map_bpm_from_beats_uses_median():
    # 120 BPM = 0.5 s per beat, with one spurious beat that would break a mean.
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 8.0])
    assert TempoMap.from_beats(beats).bpm == pytest.approx(120.0)


def test_tempo_map_without_enough_beats_falls_back():
    tempo_map = TempoMap.from_beats(np.array([1.0]))
    assert tempo_map.bpm == pytest.approx(120.0)
    assert len(tempo_map.downbeats) == 0


def test_tempo_map_constant_has_no_beats():
    tempo_map = TempoMap.constant(90.0)
    assert tempo_map.bpm == pytest.approx(90.0)
    assert len(tempo_map.beats) == 0
    assert tempo_map.has_grid is False


def test_tempo_map_from_beats_has_a_grid():
    assert TempoMap.from_beats(np.array([0.0, 0.5, 1.0])).has_grid is True


def test_track_defaults():
    track = Track(name="other", notes=[])
    assert track.program == 0
    assert track.is_drum is False
