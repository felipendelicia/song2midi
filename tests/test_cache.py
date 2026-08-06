import numpy as np
import pytest

from song2midi.cache import Cache
from song2midi.midi.model import Note


@pytest.fixture
def cache(tmp_path):
    return Cache(root=tmp_path / "cache", enabled=True)


def test_notes_are_computed_once_and_reused(cache):
    calls = []

    def compute():
        calls.append(1)
        return [Note(0.0, 1.0, 60, 100)]

    first = cache.notes("vocals-v1", compute)
    second = cache.notes("vocals-v1", compute)

    assert len(calls) == 1
    assert first == second == [Note(0.0, 1.0, 60, 100)]


def test_a_different_key_recomputes(cache):
    cache.notes("k1", lambda: [Note(0.0, 1.0, 60, 100)])
    result = cache.notes("k2", lambda: [Note(0.0, 1.0, 62, 100)])
    assert result[0].pitch == 62


def test_an_empty_note_list_is_cached_too(cache):
    calls = []

    def compute():
        calls.append(1)
        return []

    assert cache.notes("empty", compute) == []
    assert cache.notes("empty", compute) == []
    assert len(calls) == 1


def test_stems_round_trip_through_disk(cache):
    audio = np.random.default_rng(0).standard_normal((2, 1000)).astype(np.float32) * 0.1

    cache.stems("sep-v1", lambda: {"bass": audio})
    restored = cache.stems("sep-v1", lambda: pytest.fail("should not recompute"))

    assert set(restored) == {"bass"}
    assert restored["bass"].shape == audio.shape
    np.testing.assert_allclose(restored["bass"], audio, atol=1e-4)


def test_disabled_cache_always_recomputes(tmp_path):
    cache = Cache(root=tmp_path / "cache", enabled=False)
    calls = []

    def compute():
        calls.append(1)
        return [Note(0.0, 1.0, 60, 100)]

    cache.notes("k", compute)
    cache.notes("k", compute)
    assert len(calls) == 2


def test_disabled_cache_writes_nothing(tmp_path):
    root = tmp_path / "cache"
    Cache(root=root, enabled=False).notes("k", lambda: [Note(0.0, 1.0, 60, 100)])
    assert not root.exists()


def test_for_input_derives_root_from_file_content(tmp_path):
    a = tmp_path / "a.wav"
    a.write_bytes(b"content")
    b = tmp_path / "b.wav"
    b.write_bytes(b"content")
    c = tmp_path / "c.wav"
    c.write_bytes(b"different")

    assert Cache.for_input(a).root.name == Cache.for_input(b).root.name
    assert Cache.for_input(a).root.name != Cache.for_input(c).root.name


def test_for_input_respects_an_explicit_workdir(tmp_path):
    source = tmp_path / "a.wav"
    source.write_bytes(b"content")
    workdir = tmp_path / "elsewhere"

    assert workdir in Cache.for_input(source, workdir=workdir).root.parents
