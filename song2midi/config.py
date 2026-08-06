"""Run configuration and the stem to transcriber routing table."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_STEMS = ("vocals", "bass", "drums", "other")


@dataclass(frozen=True)
class StemRoute:
    """How a separated stem becomes a MIDI track."""

    transcriber_key: str
    program: int
    is_drum: bool = False


# A table, not a chain of ifs: adding an instrument is a one-line change.
STEM_ROUTING: dict[str, StemRoute] = {
    "vocals": StemRoute("vocals", program=53),
    "bass": StemRoute("bass", program=33),
    "drums": StemRoute("drums", program=0, is_drum=True),
    "other": StemRoute("polyphonic", program=0),
    "mix": StemRoute("polyphonic", program=0),
}


@dataclass(frozen=True)
class TranscriptionConfig:
    separate: bool = True
    stems: tuple[str, ...] = DEFAULT_STEMS
    device: str = "auto"
    quantize: str | None = None
    quantize_strength: float = 1.0
    use_cache: bool = True
    workdir: Path | None = None

    def fingerprint(self) -> str:
        payload = repr(sorted((key, str(value)) for key, value in asdict(self).items()))
        return hashlib.sha256(payload.encode()).hexdigest()[:8]
