import numpy as np
import pytest

from song2midi.transcription.monophonic import (
    MonophonicTranscriber,
    hz_to_midi_float,
    notes_from_f0,
    pyin_frame_length,
)

HOP = 0.01


def frames(count):
    return np.arange(count) * HOP


def constant_f0(midi_pitch, count):
    return np.full(count, 440.0 * 2 ** ((midi_pitch - 69) / 12))


def test_hz_to_midi_float_maps_a440_to_69():
    assert hz_to_midi_float(np.array([440.0]))[0] == pytest.approx(69.0)


def test_hz_to_midi_float_marks_non_positive_as_nan():
    assert np.isnan(hz_to_midi_float(np.array([0.0, -1.0]))).all()


@pytest.mark.parametrize("fmin", [30.0, 41.2, 80.0, 200.0])
def test_pyin_frame_holds_two_periods_of_fmin(fmin):
    """Below two periods pyin's estimates fall apart, which is what left the
    bass track nearly empty when the frame length was hardcoded to 2048."""
    frame_length = pyin_frame_length(fmin, 44100)
    assert frame_length >= 2 * 44100 / fmin
    assert frame_length & (frame_length - 1) == 0  # power of two, for the FFT


def test_pyin_frame_never_goes_below_the_default():
    assert pyin_frame_length(2000.0, 44100) == 2048


@pytest.mark.slow
def test_pyin_backend_finds_a_low_bass_note():
    sr = 44100
    t = np.arange(int(sr * 2)) / sr
    # E1 = 41.2 Hz, MIDI 28 — the lowest note on a 4-string bass.
    mono = 0.5 * sum(np.sin(2 * np.pi * 41.203 * h * t) / h for h in (1, 2, 3))
    audio = np.stack([mono, mono]).astype(np.float32)

    notes = MonophonicTranscriber(fmin=30.0, fmax=400.0, backend="pyin").transcribe(
        audio, sr
    )

    assert notes, "bass note not detected"
    assert max(notes, key=lambda n: n.duration).pitch == 28


def test_steady_pitch_becomes_one_note():
    count = 100
    notes = notes_from_f0(
        constant_f0(69, count), frames(count), np.ones(count), np.full(count, 0.5)
    )
    assert len(notes) == 1
    assert notes[0].pitch == 69
    assert notes[0].duration == pytest.approx(0.99, abs=0.02)


def test_pitch_change_splits_into_two_notes():
    f0 = np.concatenate([constant_f0(60, 50), constant_f0(64, 50)])
    notes = notes_from_f0(f0, frames(100), np.ones(100), np.full(100, 0.5))
    assert [n.pitch for n in notes] == [60, 64]


def test_unvoiced_frames_break_a_note():
    f0 = constant_f0(60, 100)
    confidence = np.ones(100)
    confidence[40:60] = 0.0  # 200 ms gap, well over max_gap
    notes = notes_from_f0(f0, frames(100), confidence, np.full(100, 0.5))
    assert [n.pitch for n in notes] == [60, 60]


def test_a_short_gap_does_not_break_a_note():
    f0 = constant_f0(60, 100)
    confidence = np.ones(100)
    confidence[50:52] = 0.0  # 20 ms, under max_gap
    notes = notes_from_f0(f0, frames(100), confidence, np.full(100, 0.5))
    assert len(notes) == 1


def test_vibrato_does_not_fragment_the_note():
    count = 200
    base = constant_f0(69, count)
    wobble = base * (1 + 0.02 * np.sin(np.linspace(0, 12 * np.pi, count)))
    notes = notes_from_f0(wobble, frames(count), np.ones(count), np.full(count, 0.5))
    assert len(notes) == 1
    assert notes[0].pitch == 69


def test_notes_shorter_than_the_minimum_are_dropped():
    f0 = constant_f0(60, 100)
    f0[50:52] = 440.0 * 2 ** ((72 - 69) / 12)  # 20 ms blip
    notes = notes_from_f0(f0, frames(100), np.ones(100), np.full(100, 0.5))
    assert 72 not in [n.pitch for n in notes]


def test_velocity_tracks_loudness_within_a_track():
    """Dynamics are relative within a track: the reference is the track's own
    95th percentile, so a quiet note next to loud ones reads as quiet."""
    count = 240
    f0 = np.concatenate([constant_f0(p, 60) for p in (60, 62, 64, 65)])
    rms = np.concatenate([np.full(60, level) for level in (1.0, 0.5, 0.2, 0.05)])
    notes = notes_from_f0(f0, frames(count), np.ones(count), rms)

    assert len(notes) == 4
    velocities = [n.velocity for n in notes]
    assert velocities == sorted(velocities, reverse=True), velocities
    assert velocities[0] > velocities[-1]


def test_no_note_is_inaudible():
    """Bass and drums used to render at velocity 1 - silent in a DAW. The
    linear amplitude/peak map put everything but the loudest note near zero."""
    count = 240
    f0 = np.concatenate([constant_f0(p, 60) for p in (60, 62, 64, 65)])
    rms = np.concatenate([np.full(60, level) for level in (1.0, 0.5, 0.2, 0.05)])
    notes = notes_from_f0(f0, frames(count), np.ones(count), rms)

    assert min(n.velocity for n in notes) >= 20


def test_all_unvoiced_yields_no_notes():
    assert notes_from_f0(constant_f0(60, 50), frames(50), np.zeros(50), np.full(50, 0.5)) == []


def test_notes_are_returned_sorted():
    f0 = np.concatenate([constant_f0(67, 40), constant_f0(60, 40)])
    notes = notes_from_f0(f0, frames(80), np.ones(80), np.full(80, 0.5))
    assert [n.start for n in notes] == sorted(n.start for n in notes)


@pytest.mark.slow
def test_pyin_backend_finds_a_sine():
    sr = 22050
    t = np.arange(sr * 2) / sr
    mono = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    audio = np.stack([mono, mono]).astype(np.float32)

    notes = MonophonicTranscriber(fmin=60.0, fmax=400.0, backend="pyin").transcribe(audio, sr)

    assert notes
    longest = max(notes, key=lambda n: n.duration)
    assert longest.pitch == 57  # A3


@pytest.mark.slow
def test_silence_yields_no_notes():
    audio = np.zeros((2, 22050), dtype=np.float32)
    assert MonophonicTranscriber(fmin=60.0, fmax=400.0).transcribe(audio, 22050) == []


def test_consecutive_notes_never_overlap():
    """A legato pitch change used to end note k a whole frame after note k+1
    started, and only --quantize repaired it."""
    f0 = np.concatenate([constant_f0(57, 100), constant_f0(59, 100)])
    notes = notes_from_f0(f0, frames(200), np.ones(200), np.full(200, 0.5))
    assert all(a.end <= b.start for a, b in zip(notes, notes[1:]))


def test_a_gap_closed_note_does_not_overlap_the_next():
    f0 = np.concatenate([constant_f0(60, 60), constant_f0(64, 60)])
    confidence = np.ones(120)
    confidence[55:60] = 0.0
    notes = notes_from_f0(f0, frames(120), confidence, np.full(120, 0.5))
    assert all(a.end <= b.start for a, b in zip(notes, notes[1:]))


def test_backend_picks_its_own_confidence_default():
    """pyin's voiced_prob and crepe's periodicity are different scales; a
    single threshold discards everything on one of them."""
    from song2midi.transcription.monophonic import (
        CREPE_CONFIDENCE_THRESHOLD,
        PYIN_CONFIDENCE_THRESHOLD,
    )

    pyin = MonophonicTranscriber(fmin=30.0, fmax=400.0, backend="pyin")
    crepe = MonophonicTranscriber(fmin=80.0, fmax=1100.0, backend="crepe")
    assert pyin.segmentation_kwargs["confidence_threshold"] == PYIN_CONFIDENCE_THRESHOLD
    assert crepe.segmentation_kwargs["confidence_threshold"] == CREPE_CONFIDENCE_THRESHOLD
    assert PYIN_CONFIDENCE_THRESHOLD < CREPE_CONFIDENCE_THRESHOLD


def test_an_explicit_threshold_still_wins():
    transcriber = MonophonicTranscriber(
        fmin=30.0, fmax=400.0, backend="pyin", confidence_threshold=0.9
    )
    assert transcriber.segmentation_kwargs["confidence_threshold"] == 0.9


@pytest.mark.slow
def test_pyin_rejects_aperiodic_noise():
    """Band-limited noise in the bass register is bleed, not a bass note. This
    is what put a G4 on the bass track of a real song."""
    import scipy.signal as signal

    sr = 44100
    noise = np.random.default_rng(0).standard_normal(sr * 3)
    b, a = signal.butter(4, [30 / (sr / 2), 400 / (sr / 2)], btype="band")
    filtered = signal.lfilter(b, a, noise)
    filtered = (filtered / np.max(np.abs(filtered))).astype(np.float32)

    notes = MonophonicTranscriber(fmin=30.0, fmax=400.0, backend="pyin").transcribe(
        np.stack([filtered, filtered]), sr
    )
    assert notes == []


@pytest.mark.slow
def test_pyin_keeps_a_very_quiet_real_note():
    """voiced_prob gates aperiodicity, not level, so soft playing survives."""
    sr = 44100
    t = np.arange(sr * 2) / sr
    mono = (
        0.003 * sum(np.sin(2 * np.pi * 55.0 * h * t) / h for h in (1, 2, 3))
    ).astype(np.float32)
    notes = MonophonicTranscriber(fmin=30.0, fmax=400.0, backend="pyin").transcribe(
        np.stack([mono, mono]), sr
    )
    assert notes
