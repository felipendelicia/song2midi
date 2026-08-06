"""Polyphonic transcription via Spotify's Basic Pitch.

Basic Pitch's public API only accepts a file path, so the incoming array is
written to a temporary WAV. The cost is one disk round-trip per stem, which is
negligible next to inference.

The backend is ONNX: basic-pitch picks its backend with a try/except at import
time, and the project's uv overrides keep TensorFlow out of the environment, so
the bundled `nmp.onnx` is what gets loaded.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from song2midi.midi.model import Note
from song2midi.transcription.base import amplitude_to_velocity, sort_notes

MIN_NOTE_DURATION = 0.01


class PolyphonicTranscriber:
    """General-purpose polyphonic transcriber.

    Used for guitars and keys, and for the whole mix when separation is
    unavailable.
    """

    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 100.0,
    ) -> None:
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length_ms = minimum_note_length_ms

    def transcribe(self, audio: NDArray[np.float32], sr: int) -> list[Note]:
        if not np.any(audio):
            return []

        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict

        with tempfile.TemporaryDirectory() as workdir:
            wav_path = Path(workdir) / "stem.wav"
            data = audio.T if audio.ndim > 1 else audio
            sf.write(str(wav_path), data, sr)

            _, _, note_events = predict(
                str(wav_path),
                ICASSP_2022_MODEL_PATH,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
                minimum_note_length=self.minimum_note_length_ms,
            )

        notes = []
        for start, end, pitch, amplitude, *_ in note_events:
            if end - start < MIN_NOTE_DURATION or not 0 <= pitch <= 127:
                continue
            notes.append(
                Note(
                    start=float(start),
                    end=float(end),
                    pitch=int(pitch),
                    velocity=amplitude_to_velocity(float(amplitude)),
                )
            )
        return sort_notes(notes)
