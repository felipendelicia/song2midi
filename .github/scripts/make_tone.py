"""Write a short synthetic WAV for the CI smoke test.

Standard library only, on purpose: it has to run on a runner where the project
environment does not exist, so the .exe is exercised without a Python 3.11 next
to it.

The signal is the same shape the project's own Basic Pitch tests use: a sum of
the first seven harmonics, which is the harmonic content the model was trained
on. A pure sine is unusually hard for it and would make the smoke test flaky.

Usage: python make_tone.py OUT.wav [seconds]
"""

from __future__ import annotations

import array
import math
import sys
import wave

SR = 44100
CHORD = (69, 73, 76)  # A4, C#5, E5 - a plain A major triad
PEAK = 0.75


def sawtooth(midi_pitch: int, frames: int, sr: int = SR) -> list[float]:
    freq = 440.0 * 2 ** ((midi_pitch - 69) / 12)
    out = [0.0] * frames
    for harmonic in range(1, 8):
        omega = 2 * math.pi * freq * harmonic / sr
        gain = 1.0 / harmonic
        for index in range(frames):
            out[index] += gain * math.sin(omega * index)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: make_tone.py OUT.wav [seconds]", file=sys.stderr)
        return 2
    path = argv[1]
    seconds = float(argv[2]) if len(argv) > 2 else 3.0
    frames = int(seconds * SR)

    mixed = [0.0] * frames
    for pitch in CHORD:
        for index, value in enumerate(sawtooth(pitch, frames)):
            mixed[index] += value

    largest = max(abs(value) for value in mixed) or 1.0
    attack = int(0.01 * SR)
    samples = array.array("h")
    for index, value in enumerate(mixed):
        envelope = min(1.0, index / attack) if attack else 1.0
        # A slow decay keeps the note under Basic Pitch's frame threshold for
        # its whole length instead of dropping out halfway.
        envelope *= 1.0 - 0.5 * index / frames
        sample = int(max(-1.0, min(1.0, value / largest * PEAK * envelope)) * 32767)
        samples.append(sample)
        samples.append(sample)  # duplicated: the loader wants stereo anyway

    with wave.open(path, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(samples.tobytes())

    print(f"wrote {path} ({frames} frames, {seconds:g}s, {SR} Hz stereo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
