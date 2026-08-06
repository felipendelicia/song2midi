"""On-disk cache keyed by input content.

Separation is most of the runtime, so a failure in a later stage must not cost
it. Each stage passes its own key, which means changing a quantisation flag does
not invalidate the stems.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from song2midi.midi.model import Note

STEM_SR = 44100
DIGEST_LENGTH = 16
CHUNK_SIZE = 1 << 20


def default_cache_root() -> Path:
    """Where stems and notes are cached when --workdir is not given.

    The platform cache directory, not a folder beside the input. A separated
    song leaves roughly 170 MB of stems behind, and the input is usually
    somebody's music library — often on a read-only mount, a network share or a
    cloud-synced folder, where writing beside it either fails outright or
    uploads gigabytes of intermediates.
    """
    import platformdirs

    return Path(platformdirs.user_cache_dir("song2midi"))


@dataclass
class Cache:
    root: Path
    enabled: bool = True
    _warned: bool = False

    @classmethod
    def for_input(
        cls,
        input_path: Path,
        workdir: Path | None = None,
        enabled: bool = True,
    ) -> Cache:
        digest = _file_digest(Path(input_path))
        base = Path(workdir) if workdir else default_cache_root()
        return cls(root=base / digest, enabled=enabled)

    def notes(self, key: str, compute: Callable[[], list[Note]]) -> list[Note]:
        path = self.root / "notes" / f"{key}.json"
        if self.enabled and path.is_file():
            return [Note(**item) for item in json.loads(path.read_text())]

        result = compute()
        if self.enabled:
            payload = json.dumps(
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
            self._store(lambda: (path.parent.mkdir(parents=True, exist_ok=True),
                                 path.write_text(payload)))
        return result

    def _store(self, write) -> None:
        """Persist, or carry on without a cache.

        Caching is an optimisation. A full disk or an unwritable cache
        directory must not lose a transcription that already succeeded.
        """
        try:
            write()
        except OSError as exc:
            if not self._warned:
                print(
                    f"song2midi: could not write the cache at {self.root} ({exc}); "
                    f"continuing without it",
                    file=sys.stderr,
                )
                self._warned = True

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
            # Publish the whole set at once. Writing the wavs straight into the
            # final directory means an interrupted run - Ctrl-C, an OOM, a
            # closed laptop - leaves some of the stems there, and the next run
            # finds a non-empty directory and silently transcribes a partial
            # song. Staging in a sibling and renaming makes the cache entry
            # appear only when it is complete.
            staging = directory.with_name(f"{directory.name}.partial")

            def publish():
                _remove_tree(staging)
                staging.mkdir(parents=True, exist_ok=True)
                for name, audio in result.items():
                    data = audio.T if audio.ndim > 1 else audio
                    sf.write(str(staging / f"{name}.wav"), data, STEM_SR)
                try:
                    staging.replace(directory)
                except OSError:
                    # Another process published first, or the rename is not
                    # allowed here. Their copy is as good as ours.
                    _remove_tree(staging)

            self._store(publish)
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


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()[:DIGEST_LENGTH]
