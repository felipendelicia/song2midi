import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from song2midi.config import TranscriptionConfig
from song2midi.midi.model import Note
from song2midi.pipeline import run


class StubTranscriber:
    """Returns one fixed note so the pipeline can be exercised without models."""

    def transcribe(self, audio, sr):
        return [Note(0.0, 0.5, 60, 100)]


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "in.wav"
    sf.write(str(path), np.zeros(44100, dtype=np.float32), 44100)
    return path


def test_produces_a_midi_file(tmp_path, wav, monkeypatch):
    monkeypatch.setattr(
        "song2midi.pipeline.build_transcriber", lambda key, budget=None: StubTranscriber()
    )
    out = tmp_path / "out.mid"

    result = run(wav, out, TranscriptionConfig(separate=False, use_cache=False))

    assert result == out
    assert pretty_midi.PrettyMIDI(str(out)).instruments[0].notes[0].pitch == 60


def test_default_output_path_sits_next_to_input(tmp_path, wav, monkeypatch):
    monkeypatch.setattr(
        "song2midi.pipeline.build_transcriber", lambda key, budget=None: StubTranscriber()
    )

    result = run(wav, None, TranscriptionConfig(separate=False, use_cache=False))

    assert result == wav.with_suffix(".mid")
