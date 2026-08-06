"""Cross-platform contracts.

Every test here runs identically on Linux, Windows and macOS: they exercise the
abstraction — a memory probe, a resolved executable path, a stream contract —
rather than one platform's implementation of it. That is deliberate: a test
that only runs on Windows would never run in this repo's CI matrix on the day
it matters.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

from song2midi import device
from song2midi.audio import io as audio_io
from song2midi.cli import configure_output_encoding

# --------------------------------------------------------------------------
# Available RAM
# --------------------------------------------------------------------------


def test_available_ram_is_positive_on_this_machine():
    assert device._available_ram_gb() > 0.0


def test_available_ram_matches_psutil():
    import psutil

    expected = psutil.virtual_memory().available / 1024**3
    assert device._available_ram_gb() == pytest.approx(expected, rel=0.5)


def test_available_ram_falls_back_when_psutil_cannot_answer(monkeypatch):
    """The old /proc/meminfo probe returned the fallback on Windows and macOS
    without saying so, which capped Demucs at its smallest segment."""

    def explode():
        raise RuntimeError("no memory info on this platform")

    monkeypatch.setattr("psutil.virtual_memory", explode)
    assert device._available_ram_gb() == device.FALLBACK_AVAILABLE_GB


def test_a_large_memory_machine_gets_the_largest_segment(monkeypatch):
    class Memory:
        available = 32 * 1024**3

    monkeypatch.setattr("psutil.virtual_memory", lambda: Memory())
    budget = device.resolve("cpu")
    assert budget.segment_seconds == max(device.SEGMENT_COST_GB["cpu"])


def test_a_small_memory_machine_gets_the_smallest_segment(monkeypatch):
    class Memory:
        available = 1 * 1024**3

    monkeypatch.setattr("psutil.virtual_memory", lambda: Memory())
    budget = device.resolve("cpu")
    assert budget.segment_seconds == min(device.SEGMENT_COST_GB["cpu"])


# --------------------------------------------------------------------------
# ffmpeg resolution
# --------------------------------------------------------------------------


def test_ffmpeg_executable_returns_a_path_or_none():
    resolved = audio_io.ffmpeg_executable()
    assert resolved is None or Path(resolved).is_file()


def test_ffmpeg_executable_is_absolute_when_found():
    """A bare name is resolved differently by `which` and by the process
    launcher, so anything short of a full path is a guard that can pass while
    the launch fails."""
    resolved = audio_io.ffmpeg_executable()
    if resolved is None:
        pytest.skip("ffmpeg not installed")
    assert Path(resolved).is_absolute()


def test_bundled_dirs_are_empty_when_not_frozen():
    assert audio_io._bundled_search_dirs() == []


def test_a_bundled_ffmpeg_wins_over_path(monkeypatch, tmp_path):
    """In a frozen build the co-located binary is the one that must be used:
    it is not on PATH, so `shutil.which` alone would never find it."""
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    bundled = tmp_path / name
    bundled.write_bytes(b"")

    monkeypatch.setattr(audio_io, "_bundled_search_dirs", lambda: [tmp_path])
    monkeypatch.setattr(audio_io.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    assert audio_io.ffmpeg_executable() == str(bundled)


def test_falls_back_to_path_when_nothing_is_bundled(monkeypatch):
    monkeypatch.setattr(audio_io, "_bundled_search_dirs", lambda: [])
    monkeypatch.setattr(audio_io.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    assert audio_io.ffmpeg_executable() == "/usr/bin/ffmpeg"


def test_unlaunchable_ffmpeg_becomes_a_domain_error(monkeypatch, tmp_path):
    """A file that exists but cannot execute raises OSError, not
    CalledProcessError — on Windows that is WinError 193 for a wrong-architecture
    binary. It must not escape as a raw traceback."""
    source = tmp_path / "x.m4a"
    source.write_bytes(b"not audio")
    monkeypatch.setattr(audio_io, "ffmpeg_executable", lambda: str(tmp_path / "nope"))

    with pytest.raises(audio_io.UnsupportedAudioError, match="could not run ffmpeg"):
        audio_io._load_ffmpeg(source, 44100)


def test_m4a_without_ffmpeg_reports_the_missing_dependency(monkeypatch, tmp_path):
    source = tmp_path / "x.m4a"
    source.write_bytes(b"not audio")
    monkeypatch.setattr(audio_io, "ffmpeg_executable", lambda: None)

    with pytest.raises(audio_io.UnsupportedAudioError, match="requires ffmpeg"):
        audio_io.load(source)


# --------------------------------------------------------------------------
# Format routing
# --------------------------------------------------------------------------


def test_mp3_and_opus_do_not_need_ffmpeg():
    """This is what makes a standalone Windows .exe useful: pip cannot install
    ffmpeg, but the bundled libsndfile decodes both."""
    assert ".mp3" in audio_io.NATIVE_SUFFIXES
    assert ".opus" in audio_io.NATIVE_SUFFIXES
    assert ".mp3" not in audio_io.FFMPEG_SUFFIXES


def test_ffmpeg_only_formats_are_the_ones_libsndfile_really_lacks():
    import soundfile as sf

    formats = set(sf.available_formats())
    assert "MP3" in formats
    assert "OPUS" in sf.available_subtypes("OGG")
    for suffix in audio_io.FFMPEG_SUFFIXES:
        assert suffix not in audio_io.NATIVE_SUFFIXES


def test_every_fallback_suffix_is_a_native_suffix():
    assert audio_io.FFMPEG_FALLBACK_SUFFIXES <= audio_io.NATIVE_SUFFIXES


# --------------------------------------------------------------------------
# Output encoding
# --------------------------------------------------------------------------


def test_configure_output_encoding_survives_streams_without_reconfigure(monkeypatch):
    """Under a windowed frozen build sys.stdout can be None."""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    configure_output_encoding()  # must not raise


def test_configure_output_encoding_sets_utf8_and_a_lenient_handler(monkeypatch):
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    configure_output_encoding()

    assert stream.encoding.lower().replace("-", "") == "utf8"
    assert stream.errors == "backslashreplace"


def test_a_non_encodable_path_no_longer_kills_the_process(monkeypatch):
    """The success message prints a path of a file already written to disk, so
    an encoding error would fail the run after the work succeeded."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    with pytest.raises(UnicodeEncodeError):
        print("曲.mid", file=stream)

    configure_output_encoding()
    print("曲.mid", file=stream)  # must not raise
