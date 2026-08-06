"""Transcode a WAV to MP3 with libsndfile's own LAME.

Deliberately not ffmpeg: the fixture for the "mp3 needs no external decoder"
smoke test must not itself need an external decoder. Needs the project
environment (soundfile), so it runs in the build job, not on a clean runner.

Usage: python make_mp3.py IN.wav OUT.mp3
"""

from __future__ import annotations

import sys

import soundfile as sf


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: make_mp3.py IN.wav OUT.mp3", file=sys.stderr)
        return 2
    source, target = argv[1], argv[2]

    if "MP3" not in sf.available_formats():
        print(
            f"libsndfile {sf.__libsndfile_version__} cannot encode MP3",
            file=sys.stderr,
        )
        return 1

    data, sr = sf.read(source, dtype="float32", always_2d=True)
    sf.write(target, data, sr, format="MP3")
    print(f"wrote {target} ({data.shape[0]} frames, {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
