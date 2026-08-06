"""Command line entry point."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

from song2midi.config import DEFAULT_STEMS, STEM_ROUTING, TranscriptionConfig
from song2midi.errors import Song2MidiError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="song2midi",
        description="Transcribe a song into a multitrack MIDI file.",
    )
    parser.add_argument("input", type=Path, help="audio file to transcribe")
    parser.add_argument(
        "-o", "--output", type=Path, help="output .mid (default: alongside input)"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--no-separate",
        action="store_true",
        help="skip source separation and transcribe the whole mix",
    )
    parser.add_argument(
        "--stems",
        default=",".join(DEFAULT_STEMS),
        help=f"comma-separated stems to transcribe (default: {','.join(DEFAULT_STEMS)})",
    )
    parser.add_argument("--quantize", help="grid to snap onsets to, e.g. 1/16")
    parser.add_argument(
        "--quantize-strength",
        type=float,
        default=1.0,
        help="0 leaves onsets alone, 1 snaps them fully to the grid",
    )
    parser.add_argument("--workdir", type=Path, help="cache directory")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress per-stage progress"
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TranscriptionConfig:
    stems = tuple(stem.strip() for stem in args.stems.split(",") if stem.strip())
    unknown = [stem for stem in stems if stem not in STEM_ROUTING]
    if unknown:
        raise SystemExit(
            f"song2midi: unknown stem(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(sorted(STEM_ROUTING))}"
        )
    if not 0.0 <= args.quantize_strength <= 1.0:
        raise SystemExit("song2midi: --quantize-strength must be between 0 and 1")
    return TranscriptionConfig(
        separate=not args.no_separate,
        stems=stems,
        device=args.device,
        quantize=args.quantize,
        quantize_strength=args.quantize_strength,
        use_cache=not args.no_cache,
        workdir=args.workdir,
    )


def configure_output_encoding() -> None:
    """Make stdout/stderr able to encode any path we might be handed.

    A non-UTF-8 locale — every default Windows install, and the C locale
    anywhere — gives redirected streams a code page that cannot represent most
    filenames. Since the only success output is the path of a file already
    written to disk, an encoding error here would crash the process *after* the
    work succeeded. `backslashreplace` is the load-bearing half: after this call
    printing can no longer fail.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # None under a windowed build, or a plain file
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass


def silence_third_party_noise() -> None:
    """Mute warnings the user can do nothing about.

    resampy 0.4.2 (pinned by basic-pitch) imports pkg_resources and warns about
    its deprecation; the project already pins setuptools<81 to keep it working.
    huggingface_hub warns that Windows cannot make symlinks without Developer
    Mode, which only means the model cache uses more disk. Both fire on every
    single run and bury the progress output.
    """
    warnings.filterwarnings("ignore", message=".*pkg_resources.*")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main(argv: list[str] | None = None) -> int:
    configure_output_encoding()
    silence_third_party_noise()

    from song2midi.pipeline import run

    args = parse_args(argv)
    config = build_config(args)
    try:
        output = run(args.input, args.output, config, progress=not args.quiet)
    except Song2MidiError as exc:
        print(f"song2midi: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # A separating run takes minutes, so Ctrl-C is a normal way to end one.
        # Say what survived: the stem cache is the expensive part and it is
        # reused on the next run.
        print(
            "\nsong2midi: interrupted. Anything already cached is kept, so "
            "re-running resumes from there.",
            file=sys.stderr,
        )
        return 130
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
