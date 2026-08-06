import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from song2midi.audio.io import TARGET_SR, load, to_mono
from song2midi.errors import UnsupportedAudioError


def write_wav(path, data, sr):
    sf.write(str(path), data.T if data.ndim > 1 else data, sr)
    return path


def test_loads_stereo_wav_unchanged(tmp_path):
    data = np.stack([np.linspace(-0.5, 0.5, TARGET_SR), np.zeros(TARGET_SR)])
    path = write_wav(tmp_path / "a.wav", data.astype(np.float32), TARGET_SR)

    audio, sr = load(path)

    assert sr == TARGET_SR
    assert audio.shape == (2, TARGET_SR)
    assert audio.dtype == np.float32


def test_mono_input_is_duplicated_to_stereo(tmp_path):
    data = np.zeros(TARGET_SR, dtype=np.float32)
    path = write_wav(tmp_path / "m.wav", data, TARGET_SR)

    audio, _ = load(path)

    assert audio.shape == (2, TARGET_SR)


def test_resamples_to_target_rate(tmp_path):
    data = np.zeros(22050, dtype=np.float32)
    path = write_wav(tmp_path / "s.wav", data, 22050)

    audio, sr = load(path)

    assert sr == TARGET_SR
    assert audio.shape[1] == pytest.approx(TARGET_SR, rel=0.01)


def test_missing_file_raises(tmp_path):
    with pytest.raises(UnsupportedAudioError, match="not found"):
        load(tmp_path / "nope.wav")


def test_unknown_extension_raises(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("not audio")
    with pytest.raises(UnsupportedAudioError, match="Unsupported"):
        load(path)


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"definitely not RIFF data")
    with pytest.raises(UnsupportedAudioError, match="decode"):
        load(path)


def test_mp3_is_decoded_without_ffmpeg(tmp_path, monkeypatch):
    """The bundled libsndfile decodes mp3 via libmpg123.

    Pinned with ffmpeg forced out of reach, because a standalone Windows .exe
    runs on machines that have no ffmpeg at all — if this ever regresses to the
    subprocess path, the format most users feed it stops working there.
    """
    mp3 = tmp_path / "t.mp3"
    tone = np.sin(2 * np.pi * 440 * np.arange(TARGET_SR) / TARGET_SR) * 0.4
    sf.write(str(mp3), tone.astype(np.float32), TARGET_SR, format="MP3")

    monkeypatch.setattr("song2midi.audio.io.ffmpeg_executable", lambda: None)
    audio, sr = load(mp3)

    assert sr == TARGET_SR
    assert audio.shape[0] == 2
    assert audio.dtype == np.float32
    assert np.abs(audio).max() > 0.1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_only_format_round_trips(tmp_path):
    """m4a is genuinely outside libsndfile, so it exercises the ffmpeg path."""
    source = write_wav(tmp_path / "t.wav", np.zeros(TARGET_SR, dtype=np.float32), TARGET_SR)
    m4a = tmp_path / "t.m4a"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), str(m4a)], check=True)

    audio, sr = load(m4a)

    assert sr == TARGET_SR
    assert audio.shape[0] == 2
    assert audio.dtype == np.float32


def test_to_mono_averages_channels():
    audio = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    assert to_mono(audio).tolist() == [0.5, 0.5]


def test_to_mono_passes_through_1d():
    audio = np.array([0.1, 0.2], dtype=np.float32)
    assert to_mono(audio).tolist() == pytest.approx([0.1, 0.2])
