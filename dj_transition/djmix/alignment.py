"""Tempo matching, beat grids and mix-out point selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .analysis import BEATS_PER_BAR, TrackAnalysis

MAX_COMFORTABLE_STRETCH = 0.08  # warn above ~8% tempo change
STRETCH_MODES = ("auto", "a", "b")
MULTIPLIERS = ("auto", "1", "2", "0.5")


# ------------------------------------------------------------------ beat grids

def local_grid(a: TrackAnalysis, t: float, radius_beats: int = 32) -> tuple[float, float]:
    """Fit ``time = offset + period * beat_index`` on the beats around ``t``.

    A local fit is far more precise than individual tracked beats (which are quantised
    to the analysis hop) and copes with tempo drift across a whole song.
    Returns ``(period, offset)``.
    """
    i = int(np.argmin(np.abs(a.beat_times - t)))
    centre = a.beat_index[i]
    sel = np.abs(a.beat_index - centre) <= radius_beats
    if sel.sum() < 4:
        sel = slice(None)
    period, offset = np.polyfit(a.beat_index[sel], a.beat_times[sel], 1)
    return float(period), float(offset)


def grid_time(a: TrackAnalysis, beat_idx: float, near: float | None = None) -> float:
    """Time of (possibly fractional or extrapolated) grid beat ``beat_idx``."""
    if near is None:
        near = float(np.interp(beat_idx, a.beat_index, a.beat_times))
    period, offset = local_grid(a, near)
    return offset + period * beat_idx


def nearest_downbeat(a: TrackAnalysis, t: float) -> tuple[int, float]:
    """Snap ``t`` to the closest bar line. Returns (bar number, refined time)."""
    downs = a.downbeat_times
    bar = int(np.argmin(np.abs(downs - t)))
    idx = a.beat_index[a.downbeat_mask][bar]
    return bar, grid_time(a, idx, near=downs[bar])


def local_bpm(a: TrackAnalysis, t: float) -> float:
    return 60.0 / local_grid(a, t)[0]


# ---------------------------------------------------------------- tempo match

@dataclass
class TempoPlan:
    bpm_a: float
    bpm_b: float
    multiplier: float  # B's tempo ~= multiplier * A's tempo (0.5, 1 or 2)
    rate_a: float  # playback rate for A (>1 = faster)
    rate_b: float
    stretched: str  # "a", "b" or "none"
    warnings: list[str] = field(default_factory=list)

    @property
    def played_bpm_a(self) -> float:
        return self.bpm_a * self.rate_a

    @property
    def played_bpm_b(self) -> float:
        return self.bpm_b * self.rate_b

    @property
    def grid_bpm(self) -> float:
        """Tempo of the bar grid used for transition length (the faster of the two feels)."""
        return max(self.played_bpm_a, self.played_bpm_b)

    @property
    def bar_seconds(self) -> float:
        return BEATS_PER_BAR * 60.0 / self.grid_bpm

    @property
    def change_pct(self) -> float:
        rate = self.rate_a if self.stretched == "a" else self.rate_b
        return (rate - 1.0) * 100.0

    def describe(self) -> str:
        rel = {0.5: "A double-time vs B", 1.0: "1:1", 2.0: "A half-time vs B"}[self.multiplier]
        if self.stretched == "none":
            how = "no stretch needed"
        else:
            track = "A" if self.stretched == "a" else "B"
            verb = "speed up" if self.change_pct > 0 else "slow"
            how = f"{verb} {track} by {abs(self.change_pct):.1f}%"
        return (f"{self.bpm_a:.1f} -> {self.played_bpm_a:.1f} BPM (A), "
                f"{self.bpm_b:.1f} -> {self.played_bpm_b:.1f} BPM (B); {rel}; {how}")


def best_multiplier(bpm_a: float, bpm_b: float) -> float:
    return min((0.5, 1.0, 2.0), key=lambda k: abs(math.log(k * bpm_a / bpm_b)))


def plan_tempo(bpm_a: float, bpm_b: float, stretch: str = "auto", multiplier: str = "auto") -> TempoPlan:
    """Decide how to match tempos.

    ``multiplier`` relates the two tempos: "2" means A is half-time relative to B
    (e.g. 77 vs 150), "0.5" the opposite. ``stretch`` picks which track is time-stretched:
    "auto" stretches whichever needs the smaller relative change.
    """
    if stretch not in STRETCH_MODES:
        raise ValueError(f"stretch must be one of {STRETCH_MODES}")
    if multiplier not in MULTIPLIERS:
        raise ValueError(f"multiplier must be one of {MULTIPLIERS}")
    k = best_multiplier(bpm_a, bpm_b) if multiplier == "auto" else float(multiplier)

    rate_a_only = bpm_b / (k * bpm_a)  # stretch A so that k * A == B
    rate_b_only = (k * bpm_a) / bpm_b  # stretch B so that B == k * A
    if stretch == "auto":
        stretch = "a" if abs(rate_a_only - 1) <= abs(rate_b_only - 1) else "b"

    if abs(rate_a_only - 1) < 1e-3:
        plan = TempoPlan(bpm_a, bpm_b, k, 1.0, 1.0, "none")
    elif stretch == "a":
        plan = TempoPlan(bpm_a, bpm_b, k, rate_a_only, 1.0, "a")
    else:
        plan = TempoPlan(bpm_a, bpm_b, k, 1.0, rate_b_only, "b")

    if k != 1.0:
        plan.warnings.append(
            f"Tempos {bpm_a:.1f} and {bpm_b:.1f} BPM treated as half/double-time compatible "
            f"(x{k:g})."
        )
    if abs(plan.change_pct) > MAX_COMFORTABLE_STRETCH * 100:
        plan.warnings.append(
            f"Tempo change of {abs(plan.change_pct):.1f}% is above ~8% - "
            "expect audible stretching artefacts."
        )
    return plan


# ---------------------------------------------------------------- mix-out

@dataclass
class MixOutCandidate:
    time: float  # seconds in the original (unstretched) track
    bar: int  # 0-based bar number
    score: float
    reason: str

    def as_dict(self) -> dict:
        return {"time": round(self.time, 3), "bar": self.bar, "score": round(self.score, 3),
                "reason": self.reason}


def _boundary_strengths(energy: np.ndarray, window: int = 4) -> np.ndarray:
    """|energy change in dB| between the `window` bars before and after each bar line.

    Index ``i`` is the bar line in front of bar ``i``; index ``len(energy)`` is the end.
    """
    n = len(energy)
    db = 20 * np.log10(energy + 1e-3 * (energy.max() + 1e-9))  # floor at -60 dB re. peak
    out = np.zeros(n + 1)
    for i in range(1, n + 1):
        before = db[max(0, i - window):i].mean()
        after = db[i:i + window].mean() if i < n else before - 24.0  # track end = big drop
        out[i] = after - before
    return out


def _phrase_offset(strength: np.ndarray, phrase: int = 8) -> int:
    """Which bar (mod 8) phrases start on, judged by where section changes cluster."""
    mags = np.abs(strength[1:-1])
    if len(mags) < phrase:
        return 0
    sums = [mags[(np.arange(len(mags)) + 1 - o) % phrase == 0].sum() for o in range(phrase)]
    return int(np.argmax(sums))


def suggest_mix_out_points(a: TrackAnalysis, transition_seconds: float, top: int = 5,
                           search_from: float = 0.5) -> list[MixOutCandidate]:
    """Rank bar lines in the back half of ``a`` as places to start the blend.

    A good mix-out starts on a phrase boundary (multiple of 8 bars) and lets the blend
    finish exactly where a section of A ends (energy change, or the end of the track).
    ``transition_seconds`` is the blend length in A's original timeline.
    """
    complete = a.downbeat_times + a.bar_period <= a.duration + 0.05
    downs = a.downbeat_times[complete]
    n = len(downs)
    t_bars = max(1, int(math.ceil(transition_seconds / a.bar_period - 0.05)))
    strength = _boundary_strengths(a.bar_energy[complete])
    offset = _phrase_offset(strength)

    cands: list[MixOutCandidate] = []
    first = max(1, int(n * search_from))
    for m in range(first, n - t_bars + 1):
        e = m + t_bars
        end_change = strength[e]
        phrase8 = (m - offset) % 8 == 0
        phrase16 = (m - offset) % 16 == 0
        score = (min(abs(end_change), 12) / 6.0 + min(abs(strength[m]), 12) / 12.0
                 + 0.6 * phrase8 + 0.3 * phrase16 + 0.5 * m / n)
        if e == n:
            end_desc = "blend ends with the track"
        elif end_change < -20:
            end_desc = f"blend ends at bar {e + 1} where the music stops"
        else:
            end_desc = (f"blend ends at bar {e + 1} where energy "
                        f"{'drops' if end_change < 0 else 'rises'} {abs(end_change):.1f} dB")
        phrase_desc = "16-bar phrase start" if phrase16 else ("8-bar phrase start" if phrase8 else "bar line")
        cands.append(MixOutCandidate(float(downs[m]), m, score, f"bar {m + 1}: {phrase_desc}; {end_desc}"))

    if not cands:  # track shorter than the transition: just start as late as possible
        m = max(0, n - t_bars)
        return [MixOutCandidate(float(downs[m]), m, 0.0, "track is short; earliest bar that fits")]

    cands.sort(key=lambda c: c.score, reverse=True)
    picked: list[MixOutCandidate] = []
    for c in cands:
        if all(abs(c.bar - p.bar) >= 4 for p in picked):
            picked.append(c)
        if len(picked) == top:
            break
    return picked
