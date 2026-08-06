"""On-disk cache keyed by input content.

Separation is most of the runtime, so a failure in a later stage must not cost
it. Each stage passes its own key, which means changing a quantisation flag does
not invalidate the stems.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from song2midi.midi.model import Note

CACHE_DIR_NAME = ".song2midi-cache"
STEM_SR = 44100
DIGEST_LENGTH = 16
CHUNK_SIZE = 1 << 20


@dataclass
class Cache:
    root: Path
    enabled: bool = True

    @classmethod
    def for_input(
        cls,
        input_path: Path,
        workdir: Path | None = None,
        enabled: bool = True,
    ) -> Cache:
        digest = _file_digest(Path(input_path))
        base = Path(workdir) if workdir else Path(input_path).parent / CACHE_DIR_NAME
        return cls(root=base / digest, enabled=enabled)

    def notes(self, key: str, compute: Callable[[], list[Note]]) -> list[Note]:
        path = self.root / "notes" / f"{key}.json"
        if self.enabled and path.is_file():
            return [Note(**item) for item in json.loads(path.read_text())]

        result = compute()
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    [
                        {
                            "start": note.start,
                            "end": note.end,
                            "pitch": note.pitch,
                            "velocity": note.velocity,
                        }
                        for note in result
                    ]
                )
            )
        return result

    def stems(
        self, key: str, compute: Callable[[], dict[str, NDArray]]
    ) -> dict[str, NDArray]:
        directory = self.root / "stems" / key
        if self.enabled and directory.is_dir():
            files = sorted(directory.glob("*.wav"))
            if files:
                return {
                    path.stem: np.ascontiguousarray(
                        sf.read(str(path), dtype="float32", always_2d=True)[0].T
                    )
                    for path in files
                }

        result = compute()
        if self.enabled:
            directory.mkdir(parents=True, exist_ok=True)
            for name, audio in result.items():
                data = audio.T if audio.ndim > 1 else audio
                sf.write(str(directory / f"{name}.wav"), data, STEM_SR)
        return result

    def json(self, key: str, compute: Callable[[], dict]) -> dict:
        path = self.root / f"{key}.json"
        if self.enabled and path.is_file():
            return json.loads(path.read_text())

        result = compute()
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
        return result


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()[:DIGEST_LENGTH]
