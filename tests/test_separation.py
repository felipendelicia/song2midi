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


def test_set_segment_reaches_the_inner_models():
    """apply_model's `segment` only slices the mix; HTDemucs pads every chunk
    back up to `model.segment * samplerate`, so the attribute is what actually
    bounds memory. Measured: passing segment to apply_model alone left peak RSS
    flat at 1121/1142/1171 MB for 7.8/4.0/2.0 s; setting the attribute gave
    1129/844/672 MB."""
    from song2midi.separation.demucs_sep import _set_segment

    class Inner:
        segment = 7.8

    class Bag:
        def __init__(self):
            self.models = [Inner(), Inner()]

    bag = Bag()
    _set_segment(bag, 4.0)
    assert [m.segment for m in bag.models] == [4.0, 4.0]


def test_set_segment_handles_a_bare_model():
    from song2midi.separation.demucs_sep import _set_segment

    class Bare:
        segment = 7.8

    bare = Bare()
    _set_segment(bare, 2.0)
    assert bare.segment == 2.0


def test_set_segment_ignores_a_model_without_the_attribute():
    from song2midi.separation.demucs_sep import _set_segment

    class Odd:
        pass

    _set_segment(Odd(), 4.0)  # must not raise
