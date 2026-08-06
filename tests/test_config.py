from song2midi.config import STEM_ROUTING, TranscriptionConfig


def test_defaults_separate_and_do_not_quantise():
    config = TranscriptionConfig()
    assert config.separate is True
    assert config.quantize is None
    assert config.stems == ("vocals", "bass", "drums", "other")


def test_fingerprint_is_stable_and_selective():
    a = TranscriptionConfig()
    b = TranscriptionConfig()
    c = TranscriptionConfig(quantize="1/16")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_every_default_stem_has_a_route():
    for stem in TranscriptionConfig().stems:
        assert stem in STEM_ROUTING
    assert STEM_ROUTING["drums"].is_drum is True
    assert STEM_ROUTING["bass"].program == 33
