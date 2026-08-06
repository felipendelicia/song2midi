import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from song2midi.config import TranscriptionConfig
from song2midi.errors import SeparationUnavailableError
from song2midi.midi.model import Note, TempoMap
from song2midi.pipeline import run


class StubTranscriber:
    """Returns one fixed note so the pipeline can be exercised without models."""

    def transcribe(self, audio, sr):
        return [Note(0.0, 0.5, 60, 100)]


class StubSeparator:
    def separate(self, audio, sr):
        return {name: audio for name in ("vocals", "bass", "drums", "other")}


class FailingSeparator:
    def separate(self, audio, sr):
        raise SeparationUnavailableError("demucs missing")


@pytest.fixture
def stub_stages(monkeypatch):
    monkeypatch.setattr(
        "song2midi.pipeline.build_transcriber", lambda key, budget=None: StubTranscriber()
    )
    monkeypatch.setattr(
        "song2midi.pipeline.build_separator", lambda separate, budget: StubSeparator()
    )


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


def test_each_stem_becomes_a_track(tmp_path, wav, stub_stages):
    out = tmp_path / "out.mid"

    run(wav, out, TranscriptionConfig(use_cache=False))

    names = [i.name for i in pretty_midi.PrettyMIDI(str(out)).instruments]
    assert names == ["vocals", "bass", "drums", "other"]


def test_only_requested_stems_are_transcribed(tmp_path, wav, stub_stages):
    out = tmp_path / "out.mid"

    run(wav, out, TranscriptionConfig(stems=("bass",), use_cache=False))

    assert [i.name for i in pretty_midi.PrettyMIDI(str(out)).instruments] == ["bass"]


def test_drum_track_is_flagged(tmp_path, wav, stub_stages):
    out = tmp_path / "out.mid"

    run(wav, out, TranscriptionConfig(use_cache=False))

    drums = next(
        i for i in pretty_midi.PrettyMIDI(str(out)).instruments if i.name == "drums"
    )
    assert drums.is_drum is True


def test_separation_failure_falls_back_to_the_mix(tmp_path, wav, monkeypatch, capsys):
    monkeypatch.setattr(
        "song2midi.pipeline.build_transcriber", lambda key, budget=None: StubTranscriber()
    )
    monkeypatch.setattr(
        "song2midi.pipeline.build_separator", lambda separate, budget: FailingSeparator()
    )
    out = tmp_path / "out.mid"

    run(wav, out, TranscriptionConfig(use_cache=False))

    assert [i.name for i in pretty_midi.PrettyMIDI(str(out)).instruments] == ["mix"]
    assert "demucs missing" in capsys.readouterr().err


def test_cached_run_does_not_recompute(tmp_path, wav, monkeypatch):
    calls = []

    class CountingTranscriber:
        def transcribe(self, audio, sr):
            calls.append(1)
            return [Note(0.0, 0.5, 60, 100)]

    monkeypatch.setattr(
        "song2midi.pipeline.build_transcriber",
        lambda key, budget=None: CountingTranscriber(),
    )
    config = TranscriptionConfig(separate=False, workdir=tmp_path / "cache")

    run(wav, tmp_path / "a.mid", config)
    run(wav, tmp_path / "b.mid", config)

    assert len(calls) == 1


def _stub_tempo_map(*args, **kwargs):
    return TempoMap.from_beats(np.arange(0, 4.5, 0.5))


class OffGridTranscriber:
    def transcribe(self, audio, sr):
        return [Note(0.13, 0.40, 60, 100)]


@pytest.fixture
def off_grid_stages(monkeypatch):
    monkeypatch.setattr(
        "song2midi.pipeline.build_transcriber",
        lambda key, budget=None: OffGridTranscriber(),
    )
    monkeypatch.setattr("song2midi.pipeline.detect_beats", _stub_tempo_map)


def test_detected_tempo_is_written_to_the_file(tmp_path, wav, monkeypatch):
    monkeypatch.setattr(
        "song2midi.pipeline.build_transcriber", lambda key, budget=None: StubTranscriber()
    )
    monkeypatch.setattr("song2midi.pipeline.detect_beats", _stub_tempo_map)
    out = tmp_path / "out.mid"

    run(wav, out, TranscriptionConfig(separate=False, use_cache=False))

    _, tempi = pretty_midi.PrettyMIDI(str(out)).get_tempo_changes()
    assert tempi[0] == pytest.approx(120.0, abs=0.5)


def test_quantisation_snaps_onsets_when_requested(tmp_path, wav, off_grid_stages):
    out = tmp_path / "out.mid"

    run(wav, out, TranscriptionConfig(separate=False, use_cache=False, quantize="1/16"))

    note = pretty_midi.PrettyMIDI(str(out)).instruments[0].notes[0]
    assert note.start == pytest.approx(0.125, abs=2e-3)


def test_without_the_flag_onsets_are_left_alone(tmp_path, wav, off_grid_stages):
    out = tmp_path / "out.mid"

    run(wav, out, TranscriptionConfig(separate=False, use_cache=False))

    note = pretty_midi.PrettyMIDI(str(out)).instruments[0].notes[0]
    assert note.start == pytest.approx(0.13, abs=2e-3)


def test_partial_strength_moves_onsets_only_part_way(tmp_path, wav, off_grid_stages):
    out = tmp_path / "out.mid"

    run(
        wav,
        out,
        TranscriptionConfig(
            separate=False, use_cache=False, quantize="1/16", quantize_strength=0.5
        ),
    )

    note = pretty_midi.PrettyMIDI(str(out)).instruments[0].notes[0]
    assert note.start == pytest.approx(0.1275, abs=2e-3)
