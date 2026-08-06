"""Serialisation of tracks to a Standard MIDI File."""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from song2midi.midi.model import DEFAULT_BPM, TempoMap, Track


def write_midi(
    tracks: list[Track],
    path: Path,
    tempo_map: TempoMap | None = None,
) -> Path:
    """Write tracks to `path` atomically and return the path.

    Writing to a temporary file and renaming means a crash mid-write never
    leaves a half-valid .mid behind.
    """
    path = Path(path)
    bpm = tempo_map.bpm if tempo_map is not None else DEFAULT_BPM
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    for track in tracks:
        instrument = pretty_midi.Instrument(
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
        )
        for note in track.notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
            )
        midi.instruments.append(instrument)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    midi.write(str(tmp))
    tmp.replace(path)
    return path
