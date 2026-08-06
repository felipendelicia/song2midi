import pytest

from song2midi.cli import build_config, parse_args


def test_no_separate_flag_disables_separation():
    assert build_config(parse_args(["song.mp3", "--no-separate"])).separate is False


def test_stems_are_parsed_as_a_tuple():
    config = build_config(parse_args(["song.mp3", "--stems", "bass,drums"]))
    assert config.stems == ("bass", "drums")


def test_unknown_stem_is_rejected():
    with pytest.raises(SystemExit):
        build_config(parse_args(["song.mp3", "--stems", "kazoo"]))


def test_quantize_defaults_to_full_strength():
    config = build_config(parse_args(["song.mp3", "--quantize", "1/8"]))
    assert config.quantize == "1/8"
    assert config.quantize_strength == pytest.approx(1.0)


def test_out_of_range_strength_is_rejected():
    with pytest.raises(SystemExit):
        build_config(parse_args(["song.mp3", "--quantize-strength", "1.5"]))
