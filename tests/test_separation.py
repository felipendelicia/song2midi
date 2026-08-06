import numpy as np
import pytest

from song2midi.separation.base import PassthroughSeparator

SR = 44100


def test_passthrough_returns_the_mix_under_a_single_key():
    audio = np.zeros((2, SR), dtype=np.float32)
    stems = PassthroughSeparator().separate(audio, SR)
    assert list(stems) == ["mix"]
    assert stems["mix"] is audio


def test_build_separator_honours_the_flag():
    from song2midi.device import resolve
    from song2midi.separation.demucs_sep import DemucsSeparator, build_separator

    budget = resolve("cpu")
    assert isinstance(build_separator(False, budget), PassthroughSeparator)
    assert isinstance(build_separator(True, budget), DemucsSeparator)


@pytest.mark.slow
def test_demucs_returns_four_named_stems():
    from song2midi.device import resolve
    from song2midi.separation.demucs_sep import DemucsSeparator

    t = np.arange(SR * 5) / SR
    mono = (np.sin(2 * np.pi * 110 * t) + np.sin(2 * np.pi * 660 * t)) * 0.4
    audio = np.stack([mono, mono]).astype(np.float32)

    stems = DemucsSeparator(resolve("cpu")).separate(audio, SR)

    assert set(stems) == {"drums", "bass", "other", "vocals"}
    for stem in stems.values():
        assert stem.shape == audio.shape
        assert stem.dtype == np.float32
