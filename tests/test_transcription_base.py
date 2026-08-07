import numpy as np
import pytest

from song2midi.midi.model import Note
from song2midi.transcription.base import (
    amplitude_to_velocity,
    sort_notes,
    velocity_from_energy,
)


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


# --------------------------------------------------------------------------
# Velocity curve
# --------------------------------------------------------------------------


def test_no_velocity_is_inaudible():
    """The bass and drum tracks used to render at a median velocity of 1.

    The cause was a linear amplitude/peak map: dividing by the single loudest
    event pushes everything else to the bottom of the range.
    """
    energies = [1.0, 0.5, 0.2, 0.05, 0.01, 0.3, 0.7, 0.4]
    assert min(velocity_from_energy(energies)) >= 20


def test_velocity_is_monotone_in_energy():
    energies = [0.01, 0.05, 0.2, 0.5, 1.0, 0.9, 0.7, 0.3]
    velocities = velocity_from_energy(energies)
    order = sorted(range(len(energies)), key=lambda i: energies[i])
    ranked = [velocities[i] for i in order]
    assert ranked == sorted(ranked)


def test_one_loud_outlier_does_not_flatten_the_rest():
    """A single crash cymbal must not push a whole drum track to the floor -
    the reference is the 95th percentile, not the maximum."""
    quiet_track = velocity_from_energy([0.5] * 10 + [0.25] * 10)
    with_crash = velocity_from_energy([0.5] * 10 + [0.25] * 10 + [50.0])
    assert abs(int(np.median(with_crash[:20])) - int(np.median(quiet_track))) <= 10


def test_a_bleed_level_stem_stays_quiet():
    """A stem containing only bleed must not be normalised up to full scale,
    or the most spurious track in the file becomes the loudest."""
    bleed = velocity_from_energy([0.0008, 0.0004, 0.0002, 0.001, 0.0006, 0.0003])
    real = velocity_from_energy([0.8, 0.4, 0.2, 1.0, 0.6, 0.3])
    assert max(bleed) < min(real)


def test_flat_material_gets_a_flat_velocity():
    """Fabricated dynamics cost a per-note fix in a DAW; flat costs one bulk
    edit."""
    velocities = velocity_from_energy([0.5, 0.52, 0.48, 0.51, 0.49, 0.5, 0.5, 0.5])
    assert len(set(velocities.tolist())) == 1


def test_empty_input_is_handled():
    assert len(velocity_from_energy([])) == 0


def test_all_silent_input_does_not_divide_by_zero():
    velocities = velocity_from_energy([0.0, 0.0, 0.0, 0.0])
    assert all(1 <= v <= 127 for v in velocities)
