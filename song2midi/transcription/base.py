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
