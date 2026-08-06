import numpy as np
import pytest

from song2midi.analysis.beats import detect, detect_librosa

SR = 22050


def click_track(bpm: float, seconds: float, sr: int = SR) -> np.ndarray:
    """Impulses at a fixed tempo — an unambiguous input for a beat tracker."""
    audio = np.zeros(int(seconds * sr), dtype=np.float32)
    period = 60.0 / bpm
    length = int(0.02 * sr)
    for index in range(int(seconds / period)):
        start = int(index * period * sr)
        if start + length < audio.size:
            audio[start : start + length] = np.exp(-np.linspace(0, 8, length)).astype(
                np.float32
            )
    return audio


def test_librosa_finds_the_tempo_of_a_click_track():
    mono = click_track(120.0, 10.0)
    tempo_map = detect_librosa(np.stack([mono, mono]), SR)
    assert tempo_map.bpm == pytest.approx(120.0, rel=0.05)
    assert tempo_map.has_grid


def test_librosa_on_silence_returns_a_gridless_map():
    tempo_map = detect_librosa(np.zeros((2, SR), dtype=np.float32), SR)
    assert tempo_map.has_grid is False


def _raise_import_error(*args, **kwargs):
    raise ImportError("beat_this not installed")


def test_detect_falls_back_to_librosa_when_beat_this_is_missing(monkeypatch, capsys):
    monkeypatch.setattr("song2midi.analysis.beats._detect_beat_this", _raise_import_error)
    mono = click_track(120.0, 10.0)

    tempo_map = detect(np.stack([mono, mono]), SR)

    assert tempo_map.bpm == pytest.approx(120.0, rel=0.05)
    assert len(tempo_map.downbeats) == 0  # librosa gives no downbeats
    assert "falling back to librosa" in capsys.readouterr().err


@pytest.mark.slow
def test_beat_this_finds_the_tempo():
    mono = click_track(120.0, 12.0)
    tempo_map = detect(np.stack([mono, mono]), SR)
    assert tempo_map.bpm == pytest.approx(120.0, rel=0.1)
