"""Domain exceptions for song2midi."""


class Song2MidiError(Exception):
    """Base class for all song2midi errors."""


class UnsupportedAudioError(Song2MidiError):
    """The input file cannot be decoded."""


class SeparationUnavailableError(Song2MidiError):
    """Source separation could not run; the caller should fall back."""
