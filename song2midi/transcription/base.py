"""The transcription boundary.

Everything downstream of a Transcriber works on `list[Note]`, which is why
quantisation, tempo handling and MIDI writing need no model to be tested.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from song2midi.midi.model import Note


class Transcriber(Protocol):
    """Turns audio into notes with absolute timing in seconds.

    Implementations know nothing about tempo, MIDI files or stems.
    """

    def transcribe(self, audio: NDArray[np.float32], sr: int) -> list[Note]: ...


def amplitude_to_velocity(amplitude: float) -> int:
    """Map a 0..1 amplitude to the 1..127 MIDI velocity range."""
    return int(np.clip(round(amplitude * 127), 1, 127))


def sort_notes(notes: list[Note]) -> list[Note]:
    return sorted(notes, key=lambda note: (note.start, note.pitch))


SILENT_DBFS = -60.0
LOUD_DBFS = -30.0
FLOOR_DB = -40.0
VELOCITY_FLOOR = 20
VELOCITY_CEILING = 120
# Below this a 95th/5th percentile pair is degenerate, not informative. It is
# deliberately low: the dynamic-range check below is what decides whether a
# track has dynamics, and a higher count here would short-circuit it and flatten
# genuinely dynamic short passages.
MIN_EVENTS_FOR_DYNAMICS = 3
MIN_DYNAMIC_RATIO = 2.0


def velocity_from_energy(
    energy,
    *,
    floor_db: float = FLOOR_DB,
    lo: int = VELOCITY_FLOOR,
    hi: int = VELOCITY_CEILING,
) -> NDArray[np.int64]:
    """Map per-note energies — amplitudes, not model saliences — to velocities.

    dB, not linear. A linear amplitude/peak map puts everything except the
    single loudest note near zero, which is why the bass and drum tracks used
    to render at velocity 1: inaudible in a DAW. The reference is the 95th
    percentile rather than the maximum, so one crash cannot flatten a whole
    track.

    Two guards. `gain` keeps a stem whose loudest content is only at bleed
    level quiet, instead of normalising pure noise up to full scale — without
    it the most spurious track in the file becomes the loudest. The
    dynamic-range check returns a constant when the material genuinely has no
    dynamics, rather than stretching 2% of spread across 100 velocity units:
    fabricated dynamics cost a per-note fix in a DAW, flat costs one bulk edit.

    Do NOT pass a model activation here. Basic Pitch's `amplitude` is a mean
    frame posterior, so a dB conversion of it is dimensionally meaningless.
    """
    values = np.asarray(energy, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)

    reference = float(np.percentile(values, 95))
    if reference <= 0.0:
        return np.full(values.size, lo, dtype=np.int64)

    gain = float(
        np.clip(
            (20.0 * np.log10(reference) - SILENT_DBFS) / (LOUD_DBFS - SILENT_DBFS),
            0.0,
            1.0,
        )
    )
    top = lo + gain * (hi - lo)

    quiet = float(np.percentile(values, 5))
    if (
        values.size < MIN_EVENTS_FOR_DYNAMICS
        or quiet <= 0.0
        or reference / quiet < MIN_DYNAMIC_RATIO
    ):
        return np.full(values.size, int(round((lo + top) / 2)), dtype=np.int64)

    db = 20.0 * np.log10(np.maximum(values, 1e-9) / reference)
    fraction = np.clip((db - floor_db) / -floor_db, 0.0, 1.0)
    return np.clip(np.round(lo + fraction * (top - lo)), 1, 127).astype(np.int64)
