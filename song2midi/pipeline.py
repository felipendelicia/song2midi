"""Stage orchestration.

Stage failures degrade the result rather than aborting the run: the only
unrecoverable stage is loading the input.
"""

from __future__ import annotations

import sys
from pathlib import Path

from song2midi.audio.io import load
from song2midi.cache import Cache
from song2midi.config import STEM_ROUTING, TranscriptionConfig
from song2midi.device import DeviceBudget, resolve
from song2midi.errors import SeparationUnavailableError
from song2midi.midi.model import TempoMap, Track
from song2midi.midi.writer import write_midi
from song2midi.separation.base import PassthroughSeparator
from song2midi.separation.demucs_sep import build_separator
from song2midi.transcription.base import Transcriber

SEPARATION_CACHE_KEY = "htdemucs"


def build_transcriber(key: str, budget: DeviceBudget | None = None) -> Transcriber:
    """Resolve a routing key to a transcriber.

    Imports are local so a run that uses one transcriber does not pay for the
    others' import time.
    """
    device = budget.device if budget else "cpu"

    if key == "polyphonic":
        from song2midi.transcription.polyphonic import PolyphonicTranscriber

        return PolyphonicTranscriber()
    if key == "vocals":
        from song2midi.transcription.monophonic import MonophonicTranscriber

        # crepe handles the wide range and inharmonic timbre of a voice better
        # than pyin does.
        return MonophonicTranscriber(
            fmin=80.0, fmax=1100.0, backend="crepe", device=device
        )
    if key == "bass":
        from song2midi.transcription.monophonic import MonophonicTranscriber

        # pyin needs no model and is precise in the low register.
        return MonophonicTranscriber(fmin=30.0, fmax=400.0, backend="pyin")
    if key == "drums":
        from song2midi.transcription.drums import DrumTranscriber

        return DrumTranscriber()
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
    cache = Cache.for_input(input_path, config.workdir, config.use_cache)
    budget = resolve(config.device)

    stems = _separate(audio, sr, config, budget, cache)
    tempo_map = TempoMap.constant()

    tracks = []
    for name in _stems_to_transcribe(stems, config):
        route = STEM_ROUTING[name]
        transcriber = build_transcriber(route.transcriber_key, budget)
        notes = cache.notes(
            f"{name}-{route.transcriber_key}",
            lambda t=transcriber, stem=stems[name]: t.transcribe(stem, sr),
        )
        tracks.append(
            Track(name=name, notes=notes, program=route.program, is_drum=route.is_drum)
        )

    return write_midi(tracks, output_path, tempo_map)


def _separate(audio, sr, config: TranscriptionConfig, budget, cache: Cache) -> dict:
    if not config.separate:
        return PassthroughSeparator().separate(audio, sr)

    separator = build_separator(True, budget)
    try:
        return cache.stems(
            SEPARATION_CACHE_KEY, lambda: separator.separate(audio, sr)
        )
    except SeparationUnavailableError as exc:
        warn(f"{exc}; falling back to transcribing the full mix")
        return PassthroughSeparator().separate(audio, sr)


def _stems_to_transcribe(stems: dict, config: TranscriptionConfig) -> list[str]:
    if list(stems) == ["mix"]:
        return ["mix"]
    return [name for name in config.stems if name in stems]
