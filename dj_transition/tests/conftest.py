"""Short generated test tones: kick-like pulses on a beat grid plus sustained chords."""

import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SR = 44100


def make_track(bpm: float, seconds: float, chord_midi=(60, 64, 67), sr: int = SR) -> np.ndarray:
    """Stereo test signal: pitched-down 'kick' on every beat (louder on downbeats),
    a hi-hat tick on off-beats and a chord pad so key detection has something to find."""
    n = int(seconds * sr)
    y = np.zeros(n)
    beat = 60.0 / bpm
    k_len = int(0.15 * sr)
    t_k = np.arange(k_len) / sr
    kick = np.sin(2 * np.pi * np.cumsum(50 + 100 * np.exp(-t_k * 40)) / sr) * np.exp(-t_k * 25)
    rng = np.random.default_rng(0)
    kick += 0.5 * rng.standard_normal(k_len) * np.exp(-t_k / 0.003)  # beater click
    tick = rng.standard_normal(int(0.02 * sr)) * np.exp(-np.arange(int(0.02 * sr)) / (0.004 * sr))
    i = 0
    while i * beat < seconds:
        s = int(i * beat * sr)
        amp = 1.0 if i % 4 == 0 else 0.55
        seg = kick[: n - s] * amp
        y[s:s + len(seg)] += seg
        h = int((i + 0.5) * beat * sr)
        if h < n:
            seg = tick[: n - h] * 0.04
            y[h:h + len(seg)] += seg
        i += 1
    t = np.arange(n) / sr
    for m in chord_midi:
        f = 440 * 2 ** ((m - 69) / 12)
        y += 0.08 * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))
    y = 0.8 * y / np.max(np.abs(y))
    return np.stack([y, y]).astype(np.float32)


@pytest.fixture(scope="session")
def tone_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("tones")
    a = d / "a_120.wav"
    b = d / "b_124.wav"
    sf.write(a, make_track(120, 40, (57, 60, 64)).T, SR)  # A minor triad
    sf.write(b, make_track(124, 40, (64, 67, 71)).T, SR)  # E minor triad
    return str(a), str(b)


@pytest.fixture(scope="session")
def session(tone_files):
    from djmix.engine import MixSession

    return MixSession(*tone_files, use_madmom=False, log=lambda *a: None)
