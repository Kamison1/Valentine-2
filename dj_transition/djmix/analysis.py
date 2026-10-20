"""Track analysis: tempo, beats, downbeats, key and Camelot code."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import BeatDetectionError
from .audio_io import SAMPLE_RATE, to_mono

ANALYSIS_SR = 22050
HOP = 512
BEATS_PER_BAR = 4

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler key profiles.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# --------------------------------------------------------------------------- keys

@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0..11 (C = 0)
    minor: bool

    @property
    def name(self) -> str:
        return PITCH_NAMES[self.tonic] + ("m" if self.minor else "")

    @property
    def long_name(self) -> str:
        return f"{PITCH_NAMES[self.tonic]} {'minor' if self.minor else 'major'}"

    @property
    def camelot(self) -> str:
        return camelot_code(self.tonic, self.minor)

    def transposed(self, semitones: int) -> "Key":
        return Key((self.tonic + semitones) % 12, self.minor)


def camelot_code(tonic: int, minor: bool) -> str:
    """C major = 8B, A minor = 8A; each step of a fifth moves one number clockwise."""
    major_tonic = (tonic + 3) % 12 if minor else tonic
    number = (7 * major_tonic + 7) % 12 + 1
    return f"{number}{'A' if minor else 'B'}"


def parse_camelot(code: str) -> tuple[int, str]:
    return int(code[:-1]), code[-1]


def camelot_compatible(code_a: str, code_b: str) -> bool:
    """Same code, relative major/minor (same number), or +/-1 on the wheel (same letter)."""
    na, la = parse_camelot(code_a)
    nb, lb = parse_camelot(code_b)
    if na == nb:
        return True
    return la == lb and (na - nb) % 12 in (1, 11)


def suggest_pitch_fix(key_a: Key, key_b: Key, max_shift: int = 3) -> int | None:
    """Smallest semitone shift for B that makes it harmonically compatible with A.

    Returns 0 if already compatible, ``None`` if nothing within ``max_shift`` works.
    """
    for shift in sorted(range(-max_shift, max_shift + 1), key=lambda s: (abs(s), s)):
        if camelot_compatible(key_a.camelot, key_b.transposed(shift).camelot):
            return shift
    return None


def detect_key(chroma: np.ndarray) -> tuple[Key, float]:
    """Correlate the average chroma vector against all 24 rotated key profiles."""
    profile = chroma.mean(axis=1)
    if not np.any(profile > 0):
        return Key(0, False), 0.0
    scores = []
    for minor, template in ((False, _MAJOR_PROFILE), (True, _MINOR_PROFILE)):
        for tonic in range(12):
            r = np.corrcoef(profile, np.roll(template, tonic))[0, 1]
            scores.append((r, Key(tonic, minor)))
    scores.sort(key=lambda s: s[0], reverse=True)
    confidence = float(scores[0][0] - scores[1][0])
    return scores[0][1], confidence


# ---------------------------------------------------------------------- analysis

@dataclass
class TrackAnalysis:
    name: str
    duration: float
    bpm: float
    beat_times: np.ndarray  # tracked beat positions (s)
    beat_index: np.ndarray  # position of each tracked beat on a regular grid (ints)
    downbeat_phase: int  # beats with (beat_index - phase) % 4 == 0 are downbeats
    key: Key
    key_confidence: float
    rms_db: float  # loudness of the louder half of the track
    bar_energy: np.ndarray = field(repr=False)  # RMS per bar, aligned with downbeat_times
    tempo_stability: float = 0.0  # std of beat-grid residuals in ms
    downbeat_method: str = "heuristic"
    warnings: list[str] = field(default_factory=list)

    @property
    def downbeat_mask(self) -> np.ndarray:
        return (self.beat_index - self.downbeat_phase) % BEATS_PER_BAR == 0

    @property
    def downbeat_times(self) -> np.ndarray:
        return self.beat_times[self.downbeat_mask]

    @property
    def beat_period(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_period(self) -> float:
        return BEATS_PER_BAR * self.beat_period

    @property
    def camelot(self) -> str:
        return self.key.camelot

    def summary(self) -> dict:
        return {
            "name": self.name,
            "duration": round(self.duration, 2),
            "bpm": round(self.bpm, 2),
            "key": self.key.long_name,
            "key_short": self.key.name,
            "camelot": self.camelot,
            "key_confidence": round(self.key_confidence, 3),
            "beats": int(len(self.beat_times)),
            "bars": int(self.downbeat_mask.sum()),
            "downbeat_method": self.downbeat_method,
            "tempo_stability_ms": round(self.tempo_stability, 1),
            "warnings": list(self.warnings),
        }


def _grid_indices(beat_times: np.ndarray) -> np.ndarray:
    """Assign each tracked beat an integer index on a regular grid.

    The beat tracker occasionally skips a beat; rounding each inter-beat interval to a
    multiple of the median interval keeps the bar counting intact across gaps.
    """
    ibi = np.diff(beat_times)
    med = np.median(ibi)
    steps = np.maximum(1, np.round(ibi / med)).astype(int)
    return np.concatenate([[0], np.cumsum(steps)])


def _extend_grid(beat_times: np.ndarray, beat_index: np.ndarray, duration: float,
                 n_fit: int = 32) -> tuple[np.ndarray, np.ndarray, int]:
    """Extrapolate the beat grid into intros/outros the tracker trimmed away.

    librosa drops weak beats at the edges (e.g. a beatless pad intro); continuing the
    locally fitted grid keeps bar counting valid from the very start to the very end.
    """
    head_p, head_o = np.polyfit(beat_index[:n_fit], beat_times[:n_fit], 1)
    tail_p, tail_o = np.polyfit(beat_index[-n_fit:], beat_times[-n_fit:], 1)
    first, last = int(beat_index[0]), int(beat_index[-1])
    n_pre = int(np.floor((head_o + head_p * first) / head_p))
    pre_idx = np.arange(first - n_pre, first)
    n_post = int(np.floor((duration - (tail_o + tail_p * last)) / tail_p - 0.5))
    post_idx = np.arange(last + 1, last + 1 + max(0, n_post))
    times = np.r_[head_o + head_p * pre_idx, beat_times, tail_o + tail_p * post_idx]
    index = np.r_[pre_idx, beat_index, post_idx]
    keep = (times >= 0) & (times < duration)
    shift = -int(index[keep][0])  # how much every original index moved by
    return times[keep], index[keep] + shift, shift


def _fit_grid(beat_times: np.ndarray, beat_index: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit ``t = offset + period * index``. Returns (period, offset, residual ms)."""
    period, offset = np.polyfit(beat_index, beat_times, 1)
    resid = beat_times - (offset + period * beat_index)
    return float(period), float(offset), float(np.std(resid) * 1000)


def _downbeats_madmom(path: str | None, beat_times: np.ndarray, beat_index: np.ndarray) -> int | None:
    """Use madmom's RNN downbeat tracker if it's installed; returns the phase or None."""
    if path is None:
        return None
    try:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
    except Exception:
        return None
    try:
        act = RNNDownBeatProcessor()(path)
        result = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(act)
    except Exception:
        return None
    downs = result[result[:, 1] == 1, 0]
    if len(downs) < 4:
        return None
    votes = np.zeros(BEATS_PER_BAR)
    for t in downs:
        i = int(np.argmin(np.abs(beat_times - t)))
        if abs(beat_times[i] - t) < 0.07:
            votes[beat_index[i] % BEATS_PER_BAR] += 1
    return int(np.argmax(votes))


def _downbeats_heuristic(
    beat_frames: np.ndarray, beat_index: np.ndarray, low_onset: np.ndarray, chroma: np.ndarray
) -> int:
    """Pick the bar phase where kicks/bass hits and chord changes are strongest."""
    frames = np.clip(beat_frames, 0, len(low_onset) - 1)
    kick = low_onset[frames]
    novelty = np.r_[0, np.maximum(0, np.diff(chroma, axis=1)).sum(axis=0)]
    novelty = novelty[np.clip(frames, 0, len(novelty) - 1)]

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    score = z(kick) + 0.5 * z(novelty)
    phase_scores = [score[(beat_index % BEATS_PER_BAR) == p].mean() for p in range(BEATS_PER_BAR)]
    return int(np.argmax(phase_scores))


def analyze_track(y: np.ndarray, sr: int = SAMPLE_RATE, name: str = "track",
                  path: str | None = None, use_madmom: bool = True) -> TrackAnalysis:
    import librosa

    mono = to_mono(y)
    y22 = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
    duration = len(mono) / sr

    onset_env = librosa.onset.onset_strength(y=y22, sr=ANALYSIS_SR, hop_length=HOP)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=ANALYSIS_SR, hop_length=HOP, units="frames"
    )
    beat_frames = np.asarray(beat_frames, dtype=int)
    if len(beat_frames) < 16:
        raise BeatDetectionError(
            f"Beat detection failed for {name}: only {len(beat_frames)} beats found. "
            "The track may be too quiet, too short, or have no clear rhythm."
        )
    # The tracker always returns *something*; make sure the beats are real pulses.
    clarity = onset_env[beat_frames].mean() / (onset_env.mean() + 1e-9)
    if np.percentile(onset_env, 95) < 0.5 or clarity < 1.4:
        raise BeatDetectionError(
            f"Beat detection failed for {name}: no clear rhythmic pulse "
            f"(pulse clarity {clarity:.2f}). Beatless or ambient material can't be beat-matched."
        )
    beat_times = librosa.frames_to_time(beat_frames, sr=ANALYSIS_SR, hop_length=HOP)
    beat_index = _grid_indices(beat_times)
    period, _, resid_ms = _fit_grid(beat_times, beat_index)
    bpm = 60.0 / period
    if not 40 <= bpm <= 250:
        raise BeatDetectionError(f"Beat detection for {name} gave an implausible tempo ({bpm:.1f} BPM).")

    warnings: list[str] = []
    if resid_ms > 40:
        warnings.append(
            f"{name}: tempo is not steady (beat jitter {resid_ms:.0f} ms) - alignment may drift."
        )

    chroma = librosa.feature.chroma_cqt(y=y22, sr=ANALYSIS_SR, hop_length=HOP)
    key, key_conf = detect_key(chroma)
    if key_conf < 0.02:
        warnings.append(f"{name}: key detection is uncertain ({key.name} is a weak match).")

    phase = _downbeats_madmom(path, beat_times, beat_index) if use_madmom else None
    method = "madmom" if phase is not None else "heuristic"
    if phase is None:
        low_onset = librosa.onset.onset_strength(
            y=y22, sr=ANALYSIS_SR, hop_length=HOP, fmax=200, n_mels=8
        )
        phase = _downbeats_heuristic(beat_frames, beat_index, low_onset, chroma)

    beat_times, beat_index, shift = _extend_grid(beat_times, beat_index, duration)
    phase = (phase + shift) % BEATS_PER_BAR  # keep downbeats on the same beats
    rms = librosa.feature.rms(y=y22, hop_length=HOP)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=ANALYSIS_SR, hop_length=HOP)
    loud = np.sort(rms)[len(rms) // 2:]
    rms_db = float(20 * np.log10(np.sqrt(np.mean(loud ** 2)) + 1e-9))

    analysis = TrackAnalysis(
        name=name, duration=duration, bpm=bpm, beat_times=beat_times, beat_index=beat_index,
        downbeat_phase=phase, key=key, key_confidence=key_conf, rms_db=rms_db,
        bar_energy=np.array([]), tempo_stability=resid_ms, downbeat_method=method,
        warnings=warnings,
    )
    downs = analysis.downbeat_times
    if len(downs) < 4:
        raise BeatDetectionError(f"Could not find enough bars in {name} to plan a transition.")
    edges = np.r_[downs, downs[-1] + analysis.bar_period]
    analysis.bar_energy = np.array([
        rms[(rms_times >= a) & (rms_times < b)].mean() if np.any((rms_times >= a) & (rms_times < b)) else 0.0
        for a, b in zip(edges[:-1], edges[1:])
    ])
    return analysis
