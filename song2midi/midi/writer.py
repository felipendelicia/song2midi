"""Serialisation of tracks to a Standard MIDI File."""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from song2midi.errors import OutputUnwritableError
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

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputUnwritableError(f"Cannot create {path.parent}: {exc}") from exc

    tmp = path.with_name(path.name + ".tmp")
    try:
        midi.write(str(tmp))
        # os.replace overwrites atomically on both POSIX and Windows. Windows
        # refuses when the destination is open in another process, which is
        # exactly what happens when the previous .mid is loaded in a DAW - the
        # most likely thing a user of this tool is doing.
        tmp.replace(path)
    except PermissionError as exc:
        _discard(tmp)
        raise OutputUnwritableError(
            f"Cannot write {path}: the file is open in another program "
            f"(close it in your DAW, or pass a different -o path). {exc}"
        ) from exc
    except OSError as exc:
        _discard(tmp)
        raise OutputUnwritableError(f"Cannot write {path}: {exc}") from exc
    return path


def _discard(path: Path) -> None:
    """Never leave a .tmp behind when the publish step fails."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
