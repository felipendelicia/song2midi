import mido
import pretty_midi
import pytest

from song2midi.midi.model import Note, TempoMap, Track
from song2midi.midi.writer import write_midi


@pytest.fixture
def tracks():
    return [
        Track(
            name="bass",
            notes=[Note(0.0, 0.5, 40, 90), Note(0.5, 1.0, 45, 80)],
            program=33,
        ),
        Track(
            name="drums",
            notes=[Note(0.0, 0.1, 36, 110)],
            is_drum=True,
        ),
    ]


def test_writes_notes_round_trip(tmp_path, tracks):
    out = write_midi(tracks, tmp_path / "song.mid")
    loaded = pretty_midi.PrettyMIDI(str(out))

    assert [i.name for i in loaded.instruments] == ["bass", "drums"]
    bass = loaded.instruments[0]
    assert bass.program == 33
    assert [n.pitch for n in bass.notes] == [40, 45]
    assert bass.notes[0].start == pytest.approx(0.0, abs=1e-3)
    assert bass.notes[1].end == pytest.approx(1.0, abs=1e-3)


def test_drum_track_is_marked_as_drum(tmp_path, tracks):
    out = write_midi(tracks, tmp_path / "song.mid")
    loaded = pretty_midi.PrettyMIDI(str(out))
    assert loaded.instruments[1].is_drum is True


def test_tempo_comes_from_tempo_map(tmp_path, tracks):
    out = write_midi(tracks, tmp_path / "song.mid", TempoMap.constant(90.0))
    loaded = pretty_midi.PrettyMIDI(str(out))
    _, tempi = loaded.get_tempo_changes()
    assert tempi[0] == pytest.approx(90.0, abs=0.5)


def test_empty_track_still_reaches_the_file(tmp_path):
    """A stem with no detected notes must still appear as a named track.

    pretty_midi's reader drops note-less instruments, so this asserts against
    the raw file — which is what a DAW actually reads.
    """
    out = write_midi([Track(name="vocals", notes=[])], tmp_path / "song.mid")

    names = [
        message.name
        for track in mido.MidiFile(str(out)).tracks
        for message in track
        if message.type == "track_name"
    ]
    assert names == ["vocals"]


def test_write_leaves_no_temp_file(tmp_path, tracks):
    write_midi(tracks, tmp_path / "song.mid")
    assert [p.name for p in tmp_path.iterdir()] == ["song.mid"]
