import numpy as np
import pytest

from song2midi.midi.model import Note, TempoMap
from song2midi.midi.quantize import build_grid, parse_subdivision, quantize_notes


@pytest.fixture
def tempo_map():
    # 120 BPM: beats every 0.5 s for 4 s.
    return TempoMap.from_beats(np.arange(0, 4.5, 0.5))


def test_parse_subdivision_accepts_common_grids():
    assert parse_subdivision("1/16") == 16
    assert parse_subdivision("1/8") == 8
    assert parse_subdivision("16") == 16


@pytest.mark.parametrize("text", ["banana", "2/16", "1/0", "1/-4"])
def test_parse_subdivision_rejects_nonsense(text):
    with pytest.raises(ValueError):
        parse_subdivision(text)


def test_grid_has_four_points_per_beat_at_sixteenths(tempo_map):
    grid = build_grid(tempo_map, "1/16")
    np.testing.assert_allclose(grid[:5], [0.0, 0.125, 0.25, 0.375, 0.5], atol=1e-9)


def test_grid_follows_uneven_beats():
    # A beat that drifts: the grid must follow it, not a constant BPM.
    tempo_map = TempoMap.from_beats(np.array([0.0, 0.5, 1.5]))
    grid = build_grid(tempo_map, "1/8")
    np.testing.assert_allclose(grid, [0.0, 0.25, 0.5, 1.0, 1.5], atol=1e-9)


def test_full_strength_snaps_onsets_to_the_grid(tempo_map):
    result = quantize_notes([Note(0.13, 0.4, 60, 100)], tempo_map, "1/16", strength=1.0)
    assert result[0].start == pytest.approx(0.125)


def test_duration_is_preserved(tempo_map):
    result = quantize_notes([Note(0.13, 0.43, 60, 100)], tempo_map, "1/16", strength=1.0)
    assert result[0].duration == pytest.approx(0.30)


def test_partial_strength_interpolates(tempo_map):
    # nearest sixteenth to 0.225 is 0.25; halfway is 0.2375
    result = quantize_notes([Note(0.225, 0.5, 60, 100)], tempo_map, "1/16", strength=0.5)
    assert result[0].start == pytest.approx(0.2375)


def test_zero_strength_is_a_no_op(tempo_map):
    notes = [Note(0.13, 0.4, 60, 100)]
    assert quantize_notes(notes, tempo_map, "1/16", strength=0.0) == notes


def test_a_gridless_tempo_map_leaves_notes_untouched():
    notes = [Note(0.13, 0.4, 60, 100)]
    assert quantize_notes(notes, TempoMap.constant(), "1/16") == notes


def test_monophonic_mode_truncates_overlaps(tempo_map):
    notes = [Note(0.13, 0.60, 60, 100), Note(0.26, 0.70, 62, 100)]
    result = quantize_notes(notes, tempo_map, "1/16", strength=1.0, monophonic=True)
    assert result[0].end <= result[1].start


def test_polyphonic_mode_keeps_overlaps(tempo_map):
    notes = [Note(0.13, 0.60, 60, 100), Note(0.26, 0.70, 62, 100)]
    result = quantize_notes(notes, tempo_map, "1/16", strength=1.0, monophonic=False)
    assert result[0].end > result[1].start


def test_notes_never_move_before_zero():
    tempo_map = TempoMap.from_beats(np.arange(0.5, 4.5, 0.5))
    result = quantize_notes([Note(0.02, 0.3, 60, 100)], tempo_map, "1/16")
    assert result[0].start >= 0.0


def test_result_stays_sorted(tempo_map):
    notes = [Note(0.4, 0.6, 60, 100), Note(0.13, 0.3, 62, 100)]
    result = quantize_notes(notes, tempo_map, "1/16")
    assert [n.start for n in result] == sorted(n.start for n in result)


def test_empty_input_is_returned_unchanged(tempo_map):
    assert quantize_notes([], tempo_map, "1/16") == []
