import pytest

from song2midi.config import STEM_ROUTING
from song2midi.pipeline import build_transcriber
from song2midi.transcription.drums import DrumTranscriber
from song2midi.transcription.monophonic import MonophonicTranscriber
from song2midi.transcription.polyphonic import PolyphonicTranscriber


@pytest.mark.parametrize(
    "key,expected",
    [
        ("polyphonic", PolyphonicTranscriber),
        ("vocals", MonophonicTranscriber),
        ("bass", MonophonicTranscriber),
        ("drums", DrumTranscriber),
    ],
)
def test_build_transcriber_resolves_every_routing_key(key, expected):
    assert isinstance(build_transcriber(key), expected)


def test_every_route_in_the_table_can_be_built():
    for route in STEM_ROUTING.values():
        assert build_transcriber(route.transcriber_key) is not None


def test_bass_uses_pyin_and_a_low_range():
    transcriber = build_transcriber("bass")
    assert transcriber.backend == "pyin"
    assert transcriber.fmax <= 400


def test_vocals_use_crepe_and_a_wide_range():
    transcriber = build_transcriber("vocals")
    assert transcriber.backend == "crepe"
    assert transcriber.fmax >= 1000


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown transcriber"):
        build_transcriber("kazoo")
