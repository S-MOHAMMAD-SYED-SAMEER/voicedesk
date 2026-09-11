"""What can be wrong with audio before any provider sees it."""


class AudioError(Exception):
    """Base for everything the audio layer rejects."""


class InvalidAudio(AudioError):
    """The bytes are not the format milestone 5 accepts.

    Not a WAV, the wrong sample rate, the wrong number of channels, or samples
    that are not 16-bit. Caught at the edge so a provider is never asked to
    make sense of something this system already knows it cannot use.
    """


class AudioTooLarge(AudioError):
    """The payload is bigger than one utterance has any business being.

    Checked before the bytes are parsed, so a hostile or broken frame costs a
    length comparison rather than a decode.
    """
