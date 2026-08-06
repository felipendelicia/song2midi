import pytest

from song2midi.midi.model import Note
from song2midi.transcription.base import amplitude_to_velocity, sort_notes


@pytest.mark.parametrize(
    "amplitude,expected",
    [(0.0, 1), (-5.0, 1), (1.0, 127), (2.0, 127), (0.5, 64), (0.25, 32)],
)
def test_amplitude_to_velocity_clamps_to_midi_range(amplitude, expected):
    assert amplitude_to_velocity(amplitude) == expected


def test_sort_notes_orders_by_start_then_pitch():
    notes = [Note(1.0, 2.0, 60, 100), Note(0.5, 1.0, 64, 100), Note(0.5, 1.0, 60, 100)]
    assert [(n.start, n.pitch) for n in sort_notes(notes)] == [
        (0.5, 60),
        (0.5, 64),
        (1.0, 60),
    ]
