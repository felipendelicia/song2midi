"""Failure modes that used to end in a traceback or a wrong answer.

Every one of these was found by reviewing the Windows port, but none of them is
Windows-specific: an interrupted run, a locked output file and a CPU that runs
out of memory all happen on Linux too.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from song2midi.cache import Cache
from song2midi.cli import main
from song2midi.device import resolve
from song2midi.errors import CudaUnavailableError, OutputUnwritableError
from song2midi.midi.model import Note, Track
from song2midi.midi.writer import write_midi
from song2midi.separation.demucs_sep import _is_out_of_memory

# --------------------------------------------------------------------------
# Out-of-memory detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # CUDA
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        # The CPU allocator — the wording that made the retry dead code on the
        # one device that needs it most.
        "[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: "
        "can't allocate memory: you tried to allocate 400000000 bytes.",
        "std::bad_alloc",
        "Not enough memory to allocate the buffer",
    ],
)
def test_allocation_failures_are_recognised(message):
    assert _is_out_of_memory(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "Expected 2D tensor, got 3D",
        "shape mismatch",
        "no such file",
    ],
)
def test_other_runtime_errors_are_not_treated_as_oom(message):
    assert not _is_out_of_memory(RuntimeError(message))


# --------------------------------------------------------------------------
# Partial stem cache
# --------------------------------------------------------------------------


def test_an_interrupted_stem_write_leaves_no_reusable_cache(tmp_path):
    """Writing stems straight into the final directory meant an interrupted run
    left some of them behind, and the next run silently transcribed a partial
    song."""
    cache = Cache(root=tmp_path / "cache", enabled=True)
    audio = np.zeros((2, 1000), dtype=np.float32)

    written = {"vocals": audio, "bass": audio}

    def interrupted():
        # Simulates the separator dying after producing some stems.
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cache.stems("htdemucs", interrupted)

    # Nothing publishable must exist, so the next run recomputes.
    recomputed = []

    def compute():
        recomputed.append(1)
        return written

    result = cache.stems("htdemucs", compute)
    assert len(recomputed) == 1
    assert set(result) == {"vocals", "bass"}


def test_a_leftover_partial_directory_is_not_used(tmp_path):
    cache = Cache(root=tmp_path / "cache", enabled=True)
    audio = np.zeros((2, 1000), dtype=np.float32)

    # Simulate the debris an older interrupted run would have left.
    partial = tmp_path / "cache" / "stems" / "htdemucs.partial"
    partial.mkdir(parents=True)
    (partial / "vocals.wav").write_bytes(b"junk")

    result = cache.stems("htdemucs", lambda: {"vocals": audio, "bass": audio})
    assert set(result) == {"vocals", "bass"}

    # And the completed entry replaced the debris.
    reused = cache.stems("htdemucs", lambda: pytest.fail("should not recompute"))
    assert set(reused) == {"vocals", "bass"}


def test_a_complete_cache_entry_is_reused(tmp_path):
    cache = Cache(root=tmp_path / "cache", enabled=True)
    audio = np.zeros((2, 1000), dtype=np.float32)

    cache.stems("htdemucs", lambda: {"vocals": audio, "bass": audio})
    reused = cache.stems("htdemucs", lambda: pytest.fail("should not recompute"))

    assert set(reused) == {"vocals", "bass"}


# --------------------------------------------------------------------------
# Unwritable output
# --------------------------------------------------------------------------


def test_a_locked_output_becomes_a_domain_error(tmp_path, monkeypatch):
    """On Windows, os.replace refuses when the destination is open in another
    process — which is what a DAW does with the previous .mid."""
    target = tmp_path / "song.mid"
    tracks = [Track(name="mix", notes=[Note(0.0, 0.5, 60, 100)])]

    import pathlib

    original = pathlib.Path.replace

    def deny(self, other):
        raise PermissionError(13, "used by another process")

    monkeypatch.setattr(pathlib.Path, "replace", deny)

    with pytest.raises(OutputUnwritableError, match="open in another program"):
        write_midi(tracks, target)

    monkeypatch.setattr(pathlib.Path, "replace", original)
    # And no debris survives the failure.
    assert list(tmp_path.glob("*.tmp")) == []


def test_an_unwritable_directory_becomes_a_domain_error(tmp_path, monkeypatch):
    import pathlib

    def deny(self, *args, **kwargs):
        raise PermissionError(13, "read-only")

    monkeypatch.setattr(pathlib.Path, "mkdir", deny)

    with pytest.raises(OutputUnwritableError, match="Cannot create"):
        write_midi([Track(name="mix", notes=[])], tmp_path / "sub" / "song.mid")


def test_the_cli_reports_it_instead_of_crashing(tmp_path, monkeypatch, capsys):
    import soundfile as sf

    source = tmp_path / "in.wav"
    sf.write(str(source), np.zeros(4410, dtype=np.float32), 44100)

    def explode(*args, **kwargs):
        raise OutputUnwritableError("the file is open in another program")

    monkeypatch.setattr("song2midi.pipeline.write_midi", explode)

    code = main([str(source), "--no-separate", "--no-cache"])

    assert code == 1
    err = capsys.readouterr().err
    assert "song2midi:" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------
# Interrupted run
# --------------------------------------------------------------------------


def test_ctrl_c_exits_cleanly_with_an_explanation(tmp_path, monkeypatch, capsys):
    import soundfile as sf

    source = tmp_path / "in.wav"
    sf.write(str(source), np.zeros(4410, dtype=np.float32), 44100)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("song2midi.pipeline.load", interrupt)

    code = main([str(source), "--no-separate"])

    assert code == 130
    err = capsys.readouterr().err
    assert "interrupted" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------
# CUDA requested on a CPU build
# --------------------------------------------------------------------------


def test_requesting_cuda_on_a_cpu_build_names_the_real_cause():
    """The project installs CPU-only torch, so a machine with a perfectly good
    NVIDIA card still has no CUDA. Blaming the GPU sends the user off to
    reinstall drivers that are fine."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")

    with pytest.raises(CudaUnavailableError) as excinfo:
        resolve("cuda")

    message = str(excinfo.value)
    if torch.version.cuda is None:
        assert "CPU-only build" in message
    else:
        assert "no usable GPU" in message


# --------------------------------------------------------------------------
# Cache location and degradation
# --------------------------------------------------------------------------


def test_the_cache_does_not_land_next_to_the_input(tmp_path):
    """A separated song leaves ~170 MB of stems. Somebody's music library is
    often read-only, a network share, or cloud-synced."""
    from song2midi.cache import default_cache_root

    music = tmp_path / "Music"
    music.mkdir()
    source = music / "song.mp3"
    source.write_bytes(b"content")

    cache = Cache.for_input(source)

    assert music not in cache.root.parents
    assert default_cache_root() in cache.root.parents


def test_workdir_still_overrides_the_location(tmp_path):
    source = tmp_path / "song.mp3"
    source.write_bytes(b"content")
    workdir = tmp_path / "elsewhere"

    assert workdir in Cache.for_input(source, workdir=workdir).root.parents


def test_an_unwritable_cache_does_not_lose_the_result(tmp_path, capsys):
    """Caching is an optimisation; a full disk must not discard a transcription
    that already succeeded."""
    cache = Cache(root=tmp_path / "nope", enabled=True)
    notes = [Note(0.0, 1.0, 60, 100)]

    original = pathlib.Path.mkdir

    def deny(self, *args, **kwargs):
        raise PermissionError(13, "read-only filesystem")

    pathlib.Path.mkdir = deny
    try:
        result = cache.notes("vocals", lambda: notes)
    finally:
        pathlib.Path.mkdir = original

    assert result == notes
    assert "continuing without it" in capsys.readouterr().err


def test_the_cache_warning_is_printed_once(tmp_path, capsys):
    cache = Cache(root=tmp_path / "nope", enabled=True)
    original = pathlib.Path.mkdir

    def deny(self, *args, **kwargs):
        raise PermissionError(13, "read-only filesystem")

    pathlib.Path.mkdir = deny
    try:
        cache.notes("a", lambda: [])
        cache.notes("b", lambda: [])
    finally:
        pathlib.Path.mkdir = original

    assert capsys.readouterr().err.count("continuing without it") == 1
