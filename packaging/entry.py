"""PyInstaller entry point.

Separate from `song2midi/cli.py` because PyInstaller puts the *script's* own
directory on sys.path. Pointing it at `song2midi/cli.py` would put
`song2midi/` on the path, and every `from song2midi.x import y` would fail.

`freeze_support()` has to be the first thing that runs. On Windows,
multiprocessing spawns children by re-executing the current executable; in a
frozen build that executable is song2midi.exe, so without this call every
worker restarts the whole CLI and forks again - a process bomb. torch, joblib
and demucs all reach for multiprocessing under some configurations.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()
    from song2midi.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
