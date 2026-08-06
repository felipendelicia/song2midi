"""Assert that a file is a Standard MIDI File with real note data.

Standard library only, for the same reason as make_tone.py: it runs on a runner
where the project environment is not installed.

`--min-notes 0` turns the note assertion off and only checks the container,
which is what you want if a model ever gets swapped for one with different
recall.

Usage: python check_midi.py OUT.mid [--min-tracks N] [--min-notes N]
"""

from __future__ import annotations

import argparse
import struct
import sys


def read_varlen(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def count_note_ons(track: bytes) -> int:
    """Count note-on events with non-zero velocity, honouring running status."""
    offset = 0
    status = 0
    notes = 0
    while offset < len(track):
        _, offset = read_varlen(track, offset)
        byte = track[offset]
        if byte & 0x80:
            status = byte
            offset += 1
        # else: running status, `status` carries over and `byte` is data

        if status == 0xFF:  # meta event
            offset += 1  # type
            length, offset = read_varlen(track, offset)
            offset += length
        elif status in (0xF0, 0xF7):  # sysex
            length, offset = read_varlen(track, offset)
            offset += length
        elif 0x80 <= status <= 0xEF:
            kind = status & 0xF0
            size = 1 if kind in (0xC0, 0xD0) else 2
            if kind == 0x90 and track[offset + 1] > 0:
                notes += 1
            offset += size
        else:
            raise ValueError(f"unparsable status byte {status:#04x} at {offset}")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--min-tracks", type=int, default=2)
    parser.add_argument("--min-notes", type=int, default=1)
    args = parser.parse_args(argv)

    with open(args.path, "rb") as handle:
        data = handle.read()

    if len(data) < 14 or data[:4] != b"MThd":
        raise SystemExit(f"{args.path}: not a MIDI file (magic {data[:4]!r})")
    header_len, fmt, ntrks, division = struct.unpack(">IHHH", data[4:14])
    if header_len != 6:
        raise SystemExit(f"{args.path}: bad MThd length {header_len}")

    offset = 8 + header_len
    notes = 0
    seen = 0
    while offset < len(data):
        if offset + 8 > len(data):
            raise SystemExit(f"{args.path}: truncated track header at {offset}")
        magic, length = struct.unpack(">4sI", data[offset : offset + 8])
        if magic != b"MTrk":
            raise SystemExit(f"{args.path}: expected MTrk at {offset}, got {magic!r}")
        if offset + 8 + length > len(data):
            raise SystemExit(f"{args.path}: truncated track body at {offset}")
        try:
            notes += count_note_ons(data[offset + 8 : offset + 8 + length])
        except (IndexError, ValueError) as exc:
            raise SystemExit(f"{args.path}: corrupt track at {offset}: {exc}") from exc
        seen += 1
        offset += 8 + length

    print(
        f"{args.path}: format={fmt} tracks={seen}/{ntrks} division={division} "
        f"note_ons={notes} bytes={len(data)}"
    )

    if seen != ntrks:
        raise SystemExit(f"header claims {ntrks} tracks, found {seen}")
    if seen < args.min_tracks:
        raise SystemExit(f"expected at least {args.min_tracks} tracks, got {seen}")
    if notes < args.min_notes:
        raise SystemExit(f"expected at least {args.min_notes} notes, got {notes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
