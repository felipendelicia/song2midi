"""Grid quantisation.

A pure function over `list[Note]` and a `TempoMap`. No I/O, no models, so every
edge case is cheap to test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from song2midi.midi.model import Note, TempoMap

MIN_DURATION = 1e-3
BEAT_DENOMINATOR = 4  # beats are assumed to be quarter notes


def parse_subdivision(text: str) -> int:
    """Accept '1/16' or '16' and return the denominator."""
    cleaned = text.strip()
    if "/" in cleaned:
        numerator, _, denominator = cleaned.partition("/")
        if numerator.strip() != "1":
            raise ValueError(f"Subdivision must have numerator 1, got {text!r}")
        cleaned = denominator
    try:
        value = int(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid subdivision {text!r}; expected e.g. 1/16") from exc
    if value <= 0:
        raise ValueError(f"Subdivision must be positive, got {text!r}")
    return value


def build_grid(tempo_map: TempoMap, subdivision: str) -> NDArray[np.float64]:
    """Grid points in seconds, derived from the real beat positions.

    Beats are assumed to be quarter notes, so a 1/16 grid is four points per
    beat. Deriving the grid from the beats rather than from a constant BPM is
    what keeps the end of a human-played song aligned.
    """
    denominator = parse_subdivision(subdivision)
    points_per_beat = max(1, round(denominator / BEAT_DENOMINATOR))
    beats = tempo_map.beats
    if beats.size < 2:
        return np.array([])

    grid = []
    for start, end in zip(beats[:-1], beats[1:]):
        step = (end - start) / points_per_beat
        grid.extend(start + step * index for index in range(points_per_beat))
    grid.append(float(beats[-1]))
    return np.asarray(grid, dtype=np.float64)


def quantize_notes(
    notes: list[Note],
    tempo_map: TempoMap,
    subdivision: str = "1/16",
    strength: float = 1.0,
    monophonic: bool = False,
) -> list[Note]:
    """Snap note onsets toward the nearest grid point.

    `strength` interpolates between the original position (0.0) and the grid
    (1.0), so timing can be tightened without flattening the groove.
    """
    if not notes or strength <= 0.0 or not tempo_map.has_grid:
        return notes

    grid = build_grid(tempo_map, subdivision)
    if grid.size == 0:
        return notes

    moved = []
    for note in notes:
        target = float(grid[np.argmin(np.abs(grid - note.start))])
        start = max(0.0, note.start + strength * (target - note.start))
        moved.append(
            Note(
                start=start,
                end=start + note.duration,
                pitch=note.pitch,
                velocity=note.velocity,
            )
        )

    moved.sort(key=lambda note: (note.start, note.pitch))
    return _truncate_overlaps(moved) if monophonic else moved


def _truncate_overlaps(notes: list[Note]) -> list[Note]:
    """Keep a monophonic track monophonic after onsets move."""
    result = []
    for index, note in enumerate(notes):
        end = note.end
        if index + 1 < len(notes):
            end = min(end, notes[index + 1].start)
        if end - note.start < MIN_DURATION:
            continue
        result.append(Note(note.start, end, note.pitch, note.velocity))
    return result
