"""Time-stretching and the blend itself (crossfade, bass swap, vocal EQ).

Everything here works on an *output timeline* in seconds: ``t = 0`` is the first sample
of the merged track. Song A (after stretching) plays from 0, the blend starts at
``plan.t0`` and lasts ``plan.length`` seconds, and song B continues afterwards.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

VOCAL_BAND = (300.0, 3000.0)
FILTER_MARGIN = 1.0  # seconds of context processed around the blend for clean filter edges


@dataclass
class TransitionPlan:
    """All numbers the renderer needs; built by :class:`djmix.engine.MixSession`."""

    sr: int
    duration_a: float  # original durations (s)
    duration_b: float
    rate_a: float
    rate_b: float
    pitch_a: float  # semitones
    pitch_b: float
    mix_out_a: float  # blend start in A's original timeline (s)
    b_start: float  # point of B that lands on the mix-out downbeat (B's original timeline)
    length: float  # blend length in output seconds
    beat_sec: float
    bar_sec: float
    swap_time: float  # bass swap moment (output s)
    gain_b: float = 1.0
    bass_swap: bool = True
    crossover_hz: float = 180.0
    vocal_cut: bool = True
    vocal_cut_db: float = -6.0

    @property
    def t0(self) -> float:
        return self.mix_out_a / self.rate_a

    @property
    def t1(self) -> float:
        return self.t0 + self.length

    @property
    def total_duration(self) -> float:
        return self.t0 + (self.duration_b - self.b_start) / self.rate_b

    def a_to_out(self, t):
        return np.asarray(t) / self.rate_a

    def b_to_out(self, t):
        return (np.asarray(t) - self.b_start) / self.rate_b + self.t0


# ------------------------------------------------------------- stretch / pitch

def rubberband_available() -> bool:
    if shutil.which("rubberband") is None:
        return False
    try:
        import pyrubberband  # noqa: F401
    except ImportError:
        return False
    return True


def stretch_and_shift(y: np.ndarray, sr: int, rate: float, semitones: float = 0.0) -> np.ndarray:
    """Change tempo by ``rate`` (>1 = faster) and pitch by ``semitones`` in one pass."""
    if abs(rate - 1.0) < 1e-6 and semitones == 0:
        return y
    if rubberband_available():
        import pyrubberband as pyrb

        if abs(rate - 1.0) < 1e-6:
            out = pyrb.pitch_shift(y.T, sr, semitones)
        else:
            rbargs = {"--pitch": str(semitones)} if semitones else None
            out = pyrb.time_stretch(y.T, sr, rate, rbargs=rbargs)
        return np.ascontiguousarray(out.T, dtype=np.float32)

    import librosa

    warnings.warn("Rubber Band not found - falling back to librosa's phase vocoder (lower quality).")
    out = y
    if abs(rate - 1.0) >= 1e-6:
        out = librosa.effects.time_stretch(out, rate=rate)
    if semitones:
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=semitones)
    return np.ascontiguousarray(out, dtype=np.float32)


def _padded_slice(y: np.ndarray, start: int, stop: int) -> np.ndarray:
    """``y[:, start:stop]`` with zeros where the range falls outside the signal."""
    out = np.zeros((y.shape[0], max(0, stop - start)), dtype=np.float32)
    s0, s1 = max(start, 0), min(stop, y.shape[1])
    if s1 > s0:
        out[:, s0 - start:s1 - start] = y[:, s0:s1]
    return out


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if y.shape[1] >= n:
        return y[:, :n]
    return np.pad(y, ((0, 0), (0, n - y.shape[1])))


def stretched_segment(y: np.ndarray, sr: int, t0: float, t1: float, rate: float,
                      semitones: float = 0.0, pad: float = 0.5) -> np.ndarray:
    """Audio of the stretched track between ``t0`` and ``t1`` (stretched-timeline seconds).

    Only the needed part of the original is processed, which is what makes previews fast.
    """
    n = int(round((t1 - t0) * sr))
    if n <= 0:
        return np.zeros((y.shape[0], 0), dtype=np.float32)
    src = _padded_slice(y, int(round((t0 * rate - pad) * sr)), int(round((t1 * rate + pad) * sr)))
    out = stretch_and_shift(src, sr, rate, semitones)
    skip = int(round(pad / rate * sr))
    return _fit_length(out[:, skip:], n)


# ------------------------------------------------------------------- blend

@lru_cache(maxsize=16)
def _lowpass(sr: int, fc: float):
    return butter(4, fc, btype="lowpass", fs=sr, output="sos")


@lru_cache(maxsize=4)
def _vocal_bandpass(sr: int):
    return butter(2, VOCAL_BAND, btype="bandpass", fs=sr, output="sos")


def _ramp(x: np.ndarray) -> np.ndarray:
    """Raised-cosine 0 -> 1 ramp for x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def envelopes(t: np.ndarray, plan: TransitionPlan) -> dict[str, np.ndarray]:
    """Gain curves at output times ``t``.

    ``gain_*``: equal-power volume crossfade. ``low_*``: how much of each track's bass
    band is kept; with bass swap A's bass leaves in the beat before ``swap_time`` and
    B's arrives in the beat after, so they never overlap. ``vocal``: depth of the
    vocal-range cut applied to A.
    """
    p = np.clip((t - plan.t0) / plan.length, 0.0, 1.0)
    env = {"gain_a": np.cos(p * np.pi / 2), "gain_b": np.sin(p * np.pi / 2)}
    if plan.bass_swap:
        ramp = plan.beat_sec
        env["low_a"] = 1.0 - _ramp((t - (plan.swap_time - ramp)) / ramp)
        env["low_b"] = _ramp((t - plan.swap_time) / ramp)
    else:
        env["low_a"] = np.ones_like(t)
        env["low_b"] = np.ones_like(t)
    depth = 1.0 - 10 ** (plan.vocal_cut_db / 20.0) if plan.vocal_cut else 0.0
    env["vocal"] = depth * _ramp((t - plan.t0) / plan.bar_sec)
    return env


def _process(seg: np.ndarray, t_start: float, plan: TransitionPlan, which: str) -> np.ndarray:
    """Apply the blend envelopes for track ``which`` ('a' or 'b') to ``seg``.

    The band split is complementary (``x = low + (x - low)``), so wherever the envelopes
    are at rest the output equals the input sample-for-sample and can be spliced
    seamlessly with unprocessed audio.
    """
    if seg.shape[1] == 0:
        return seg
    t = t_start + np.arange(seg.shape[1]) / plan.sr
    env = envelopes(t, plan)
    out = seg.astype(np.float64)
    keep_low = env["low_a" if which == "a" else "low_b"]
    if np.any(keep_low < 1.0):
        low = sosfiltfilt(_lowpass(plan.sr, plan.crossover_hz), seg, axis=-1)
        out -= low * (1.0 - keep_low)
    if which == "a" and np.any(env["vocal"] > 0):
        band = sosfiltfilt(_vocal_bandpass(plan.sr), seg, axis=-1)
        out -= band * env["vocal"]
    out *= env["gain_a" if which == "a" else "gain_b"]
    return out.astype(np.float32)


def render_mix(a: np.ndarray, b: np.ndarray, plan: TransitionPlan, w0: float = 0.0,
               w1: float | None = None, return_stems: bool = False):
    """Render the output timeline between ``w0`` and ``w1`` seconds.

    Returns the stereo mix, or ``(mix, a_part, b_part)`` with ``return_stems``.
    """
    sr = plan.sr
    w1 = plan.total_duration if w1 is None else w1
    # Snap the window to whole samples so previews line up exactly with the full render.
    w0, w1 = round(w0 * sr) / sr, round(w1 * sr) / sr
    n = int(round((w1 - w0) * sr))
    out = np.zeros((2, n), dtype=np.float32)
    stems = (np.zeros_like(out), np.zeros_like(out)) if return_stems else None
    reg0, reg1 = plan.t0 - FILTER_MARGIN, plan.t1 + FILTER_MARGIN

    def add(buf: np.ndarray, start_t: float, which: str) -> None:
        i0 = int(round((start_t - w0) * sr))
        buf = buf[:, : max(0, n - i0)]
        out[:, i0:i0 + buf.shape[1]] += buf
        if stems is not None:
            stems[0 if which == "a" else 1][:, i0:i0 + buf.shape[1]] += buf

    # Song A: plays until the blend has finished.
    a_end = min(w1, reg1)
    if a_end > w0:
        seg = stretched_segment(a, sr, w0, a_end, plan.rate_a, plan.pitch_a)
        k = max(0, int(round((reg0 - w0) * sr)))
        seg[:, k:] = _process(seg[:, k:], w0 + k / sr, plan, "a")
        add(seg, w0, "a")

    # Song B: enters at the blend and carries on to the end.
    b_from = max(w0, reg0)
    if w1 > b_from:
        b_t0 = (b_from - plan.t0) + plan.b_start / plan.rate_b  # B's stretched timeline
        seg = stretched_segment(b, sr, b_t0, b_t0 + (w1 - b_from), plan.rate_b, plan.pitch_b)
        seg *= plan.gain_b
        k = int(round((min(w1, reg1) - b_from) * sr))
        seg[:, :k] = _process(seg[:, :k], b_from, plan, "b")
        add(seg, b_from, "b")

    if return_stems:
        return out, stems[0], stems[1]
    return out
