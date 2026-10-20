"""djmix - automatic DJ-style transitions between two songs.

The package is split into small modules that the CLI and the web app share:

- ``audio_io``  loading/exporting audio
- ``analysis``  BPM, beats, downbeats, key / Camelot code
- ``alignment`` tempo matching, mix-out point suggestions, beat grids
- ``mixing``    time-stretching and the actual blend (crossfade, bass swap, EQ)
- ``engine``    high-level session object used by the CLI and the web UI
- ``plotting``  matplotlib waveform plot of the blend region
"""

__version__ = "0.1.0"


class DJMixError(Exception):
    """Base class for errors that should be shown to the user verbatim."""


class AudioLoadError(DJMixError):
    """A file could not be read or decoded."""


class BeatDetectionError(DJMixError):
    """Beat tracking did not find a usable beat grid."""
