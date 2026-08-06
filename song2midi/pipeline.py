"""Stage orchestration.

Stage failures degrade the result rather than aborting the run: the only
unrecoverable stage is loading the input.
"""

from __future__ import annotations

import sys
from pathlib import Path

from song2midi.audio.io import load
from song2midi.config import STEM_ROUTING, TranscriptionConfig
from song2midi.midi.model import TempoMap, Track
from song2midi.midi.writer import write_midi
from song2midi.transcription.base import Transcriber


def build_transcriber(key: str, budget=None) -> Transcriber:
    """Resolve a routing key to a transcriber.

    Imports are local so a run that uses one transcriber does not pay for the
    others' import time.
    """
    if key == "polyphonic":
        from song2midi.transcription.polyphonic import PolyphonicTranscriber

        return PolyphonicTranscriber()
    raise ValueError(f"Unknown transcriber key: {key}")


def warn(message: str) -> None:
    print(f"song2midi: {message}", file=sys.stderr)


def run(
    input_path: Path,
    output_path: Path | None,
    config: TranscriptionConfig,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_suffix(".mid")

    audio, sr = load(input_path)
    stems = {"mix": audio}
    tempo_map = TempoMap.constant()

    tracks = []
    for name, stem_audio in stems.items():
        route = STEM_ROUTING[name]
        transcriber = build_transcriber(route.transcriber_key)
        notes = transcriber.transcribe(stem_audio, sr)
        tracks.append(
            Track(name=name, notes=notes, program=route.program, is_drum=route.is_drum)
        )

    return write_midi(tracks, output_path, tempo_map)
