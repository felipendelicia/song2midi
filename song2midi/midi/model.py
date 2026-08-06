"""Core data types.

Deliberately free of project dependencies so that every downstream stage can be
tested without loading a model or touching disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

DEFAULT_BPM = 120.0


@dataclass(frozen=True)
class Note:
    """A single note with absolute timing in seconds."""

    start: float
    end: float
    pitch: int
    velocity: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"pitch {self.pitch} outside MIDI range 0-127")
        if not 1 <= self.velocity <= 127:
            raise ValueError(f"velocity {self.velocity} outside MIDI range 1-127")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Track:
    """A named group of notes bound to a General MIDI program."""

    name: str
    notes: list[Note]
    program: int = 0
    is_drum: bool = False


@dataclass(frozen=True)
class TempoMap:
    """Beat grid of a song.

    Stores the actual beat positions rather than a single BPM: human-played
    songs drift, and quantising against a constant tempo drags the end of the
    song out of alignment.
    """

    bpm: float
    beats: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    downbeats: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_beats(
        cls,
        beats: NDArray[np.float64],
        downbeats: NDArray[np.float64] | None = None,
    ) -> TempoMap:
        beats = np.asarray(beats, dtype=np.float64)
        if beats.size < 2:
            return cls(bpm=DEFAULT_BPM, beats=beats, downbeats=np.array([]))
        # Median, not mean: robust to spurious or missed beats.
        interval = float(np.median(np.diff(beats)))
        bpm = 60.0 / interval if interval > 0 else DEFAULT_BPM
        return cls(
            bpm=bpm,
            beats=beats,
            downbeats=np.asarray(
                downbeats if downbeats is not None else [], dtype=np.float64
            ),
        )

    @classmethod
    def constant(cls, bpm: float = DEFAULT_BPM) -> TempoMap:
        """Tempo map with no beat grid. Quantisation is unavailable against it."""
        return cls(bpm=bpm, beats=np.array([]), downbeats=np.array([]))

    @property
    def has_grid(self) -> bool:
        return self.beats.size >= 2
