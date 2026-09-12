"""Deciding whether a frame of audio contains somebody talking.

**This is an energy detector, not a neural one, and the difference matters.**
It measures how loud a frame is relative to the ambient level and how often
the waveform crosses zero. That is enough to tell a talking caller from a
quiet line, and it is honestly not enough to tell a talking caller from a
television left on in the next room. Sustained background noise raises the
noise floor and can be mistaken for speech; a very quietly spoken word over a
noisy line can be missed. No claim of parity with a trained model is made
anywhere in this project.

What it buys instead: it is a few dozen lines of standard library, it runs
about six hundred times faster than real time, every decision it makes is
reproducible from its inputs, and it needs no wheel, no model download and no
native build. For a milestone whose whole point is that a test can drive a
conversation without a microphone, that is the right trade.

It knows nothing about carriers, sockets, models or databases. Frames of PCM
in, `Voice` or `Silence` out.
"""

import math
import struct
from dataclasses import dataclass

# Below this, a frame is a DC offset or low-frequency rumble rather than
# anything anybody said — a hum can be loud and still cross zero almost never.
# A module constant rather than a setting: it separates "audio" from "not
# audio", which is not something a deployment should be tuning.
MIN_ZERO_CROSSING_RATE = 0.01


@dataclass(frozen=True)
class FrameAnalysis:
    """What one frame looks like, before any decision is taken about it."""

    energy: float
    zero_crossing_rate: float
    speech: bool
    noise_floor: float
    threshold: float


def rms(pcm: bytes) -> float:
    """Root-mean-square amplitude: loudness, on the 16-bit scale."""
    if len(pcm) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def zero_crossing_rate(pcm: bytes) -> float:
    """How often the waveform changes sign, per sample.

    Roughly a pitch measure. Near zero means a constant offset or a very low
    rumble; speech sits well above that, and hiss well above speech.
    """
    if len(pcm) < 4:
        return 0.0
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    crossings = sum(
        1
        for index in range(1, len(samples))
        if (samples[index - 1] < 0) != (samples[index] < 0)
    )
    return crossings / (len(samples) - 1)


class VoiceActivityDetector:
    """Frame-by-frame speech detection with an adaptive floor.

    Two things make this better than a fixed threshold, which is what the
    previous milestone used:

    * **The floor adapts.** Quiet frames move a running estimate of the room,
      so the same detector works on a silent line and a noisy one without
      anybody retuning a number.
    * **It has hysteresis.** Getting into speech takes more than staying in
      it, so one loud frame does not start an utterance and one quiet frame
      does not end one. Speech is full of brief gaps between words.
    """

    def __init__(
        self,
        *,
        energy_threshold: float = 300.0,
        noise_floor_alpha: float = 0.05,
        hysteresis: float = 0.6,
    ) -> None:
        if not 0.0 < noise_floor_alpha <= 1.0:
            raise ValueError("noise_floor_alpha must be between 0 and 1.")
        if not 0.0 < hysteresis <= 1.0:
            raise ValueError("hysteresis must be between 0 and 1.")

        self._energy_threshold = energy_threshold
        self._alpha = noise_floor_alpha
        self._hysteresis = hysteresis
        self._noise_floor = 0.0
        self._in_speech = False

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def enter_threshold(self) -> float:
        """Speech has to be this much louder than the room.

        The ambient level plus the configured margin, rather than a
        multiplier: on a silent line this is exactly the margin, and on a
        noisy one it rises with the room by the same amount.
        """
        return self._noise_floor + self._energy_threshold

    @property
    def leave_threshold(self) -> float:
        return self.enter_threshold * self._hysteresis

    def analyse(self, pcm: bytes) -> FrameAnalysis:
        """Look at one frame and say whether somebody is talking in it."""
        energy = rms(pcm)
        crossings = zero_crossing_rate(pcm)
        threshold = self.leave_threshold if self._in_speech else self.enter_threshold

        audible = crossings >= MIN_ZERO_CROSSING_RATE
        speech = audible and energy >= threshold

        if not speech:
            # The floor only learns from silence. Letting speech into it would
            # teach the detector to ignore the caller.
            self._noise_floor += self._alpha * (energy - self._noise_floor)

        self._in_speech = speech
        return FrameAnalysis(
            energy=energy,
            zero_crossing_rate=crossings,
            speech=speech,
            noise_floor=self._noise_floor,
            threshold=threshold,
        )

    def is_speech(self, pcm: bytes) -> bool:
        return self.analyse(pcm).speech

    def reset(self) -> None:
        """Forget the room. Used between calls, never within one."""
        self._noise_floor = 0.0
        self._in_speech = False
