"""Stage orchestration.

Stage failures degrade the result rather than aborting the run: the only
unrecoverable stage is loading the input.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from song2midi.analysis.beats import detect as detect_beats
from song2midi.audio.io import load
from song2midi.cache import Cache
from song2midi.config import STEM_ROUTING, TranscriptionConfig
from song2midi.device import DeviceBudget, resolve
from song2midi.errors import SeparationUnavailableError
from song2midi.midi.model import Track
from song2midi.midi.quantize import quantize_notes
from song2midi.midi.writer import write_midi
from song2midi.separation.base import PassthroughSeparator
from song2midi.separation.demucs_sep import build_separator
from song2midi.transcription.base import Transcriber

SEPARATION_CACHE_KEY = "htdemucs"
# Tracks that can only sound one note at a time; quantisation must not leave
# them overlapping after onsets move.
MONOPHONIC_KEYS = frozenset({"vocals", "bass"})


def build_transcriber(key: str, budget: DeviceBudget | None = None) -> Transcriber:
    """Resolve a routing key to a transcriber.

    Imports are local so a run that uses one transcriber does not pay for the
    others' import time.
    """
    device = budget.device if budget else "cpu"

    if key == "polyphonic":
        from song2midi.transcription.polyphonic import PolyphonicTranscriber

        # Basic Pitch's defaults re-onset held notes: at 0.5 / 100 ms the
        # `other` stem of a 4-minute song came back with 2133 notes. 0.7 /
        # 200 ms cuts that sharply while sounding time holds - it consolidates
        # duplicate onsets rather than deleting content. 100 ms was below
        # basic-pitch's own default of 127.7 ms in the first place.
        return PolyphonicTranscriber(onset_threshold=0.7, minimum_note_length_ms=200.0)
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


class Progress:
    """Announces each stage with elapsed time.

    Separation alone runs for minutes with no output of its own, so without
    this a working run is indistinguishable from a hung one. Everything goes to
    stderr; stdout carries only the output path, so `song2midi x.mp3 > out.txt`
    still yields just the path.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started = time.monotonic()

    def __call__(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed = int(time.monotonic() - self.started)
        print(
            f"song2midi: [{elapsed // 60:02d}:{elapsed % 60:02d}] {message}",
            file=sys.stderr,
            flush=True,
        )


def run(
    input_path: Path,
    output_path: Path | None,
    config: TranscriptionConfig,
    progress: bool = True,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_suffix(".mid")
    report = Progress(enabled=progress)

    report(f"reading {input_path.name}")
    audio, sr = load(input_path)
    duration = audio.shape[-1] / sr
    report(f"{duration / 60:.1f} min of audio")

    cache = Cache.for_input(input_path, config.workdir, config.use_cache)
    budget = resolve(config.device)
    report(f"using {budget.device} ({budget.available_gb:.1f} GB free)")

    stems = _separate(audio, sr, config, budget, cache, report)

    report("detecting tempo")
    tempo_map = detect_beats(audio, sr, budget.device)
    report(f"{tempo_map.bpm:.1f} BPM")

    tracks = []
    for name in _stems_to_transcribe(stems, config):
        route = STEM_ROUTING[name]
        report(f"transcribing {name}")
        transcriber = build_transcriber(route.transcriber_key, budget)
        notes = cache.notes(
            f"{name}-{route.transcriber_key}",
            lambda t=transcriber, stem=stems[name]: t.transcribe(stem, sr),
        )
        report(f"{name}: {len(notes)} notes")
        if config.quantize:
            notes = quantize_notes(
                notes,
                tempo_map,
                config.quantize,
                config.quantize_strength,
                monophonic=route.transcriber_key in MONOPHONIC_KEYS,
            )
        tracks.append(
            Track(name=name, notes=notes, program=route.program, is_drum=route.is_drum)
        )

    report(f"writing {output_path.name}")
    return write_midi(tracks, output_path, tempo_map)


def _separate(
    audio, sr, config: TranscriptionConfig, budget, cache: Cache, report
) -> dict:
    if not config.separate:
        return PassthroughSeparator().separate(audio, sr)

    separator = build_separator(True, budget)

    def compute():
        report(
            f"separating stems with htdemucs (segment={budget.segment_seconds}s) - "
            f"the slow part, several minutes on CPU"
        )
        return separator.separate(audio, sr)

    try:
        stems = cache.stems(SEPARATION_CACHE_KEY, compute)
        report(f"stems ready: {', '.join(stems)}")
        return stems
    except SeparationUnavailableError as exc:
        warn(f"{exc}; falling back to transcribing the full mix")
        return PassthroughSeparator().separate(audio, sr)


def _stems_to_transcribe(stems: dict, config: TranscriptionConfig) -> list[str]:
    if list(stems) == ["mix"]:
        return ["mix"]
    return [name for name in config.stems if name in stems]
