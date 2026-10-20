"""Generate two synthetic demo songs for trying the tool without real music.

- song_a_moody.mp3 : ~77 BPM, A minor, pads + sub bass + "vocal" lead, sectioned
- song_b_trap.wav  : ~145 BPM, D minor, 808s, hi-hat rolls, half-time snare

Usage:  python scripts/make_samples.py [output_dir]
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 44100
rng = np.random.default_rng(7)


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def env_exp(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (decay * SR))


def kick(dur=0.35, f0=130, f1=45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    freq = f1 + (f0 - f1) * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * env_exp(n, 0.09)


def snare(dur=0.25):
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, [1, -0.6], mode="same")
    t = np.arange(n) / SR
    return (0.6 * noise + 0.5 * np.sin(2 * np.pi * 190 * t)) * env_exp(n, 0.05)


def hat(dur=0.05):
    n = int(dur * SR)
    noise = np.diff(rng.standard_normal(n + 1))
    return noise * env_exp(n, 0.012) * 0.5


def tone(freq, dur, decay=None, harmonics=(1.0,), vibrato=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    phase_mod = vibrato * np.sin(2 * np.pi * 5.0 * t)
    out = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t + phase_mod * (i + 1))
              for i, a in enumerate(harmonics))
    att = np.minimum(1, t / 0.02)
    rel = np.minimum(1, (dur - t) / 0.05)
    e = att * np.clip(rel, 0, 1)
    if decay:
        e = e * env_exp(n, decay)
    return out * e


def bass808(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = freq * (1 + 0.6 * np.exp(-t * 40))
    sig = np.tanh(2.0 * np.sin(2 * np.pi * np.cumsum(f) / SR))
    return sig * env_exp(n, 0.6) * np.clip((dur - t) / 0.03, 0, 1)


class Track:
    def __init__(self, seconds: float):
        self.buf = np.zeros((2, int(seconds * SR) + SR))

    def add(self, sig, at, gain=1.0, pan=0.0):
        i = int(at * SR)
        sig = sig[: max(0, self.buf.shape[1] - i)]
        left, right = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
        self.buf[0, i:i + len(sig)] += sig * gain * left
        self.buf[1, i:i + len(sig)] += sig * gain * right

    def finish(self, length):
        y = self.buf[:, : int(length * SR)]
        return (y / np.max(np.abs(y)) * 0.9).astype(np.float32)


def song_a(bpm=77.0):
    beat = 60 / bpm
    bar = 4 * beat
    # (name, bars, drums, lead, energy)
    sections = [("intro", 8, False, False, 0.5), ("verse", 16, True, True, 0.75),
                ("chorus", 16, True, True, 1.0), ("break", 8, False, True, 0.55),
                ("chorus", 16, True, True, 1.0), ("outro", 16, True, False, 0.6)]
    total = sum(s[1] for s in sections) * bar + 4
    tr = Track(total)
    chords = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]  # Am F C G
    roots = [45, 41, 36, 43]
    melody = [69, 72, 71, 67, 69, 64, 65, 67]
    b = 0
    for name, n_bars, drums, lead, energy in sections:
        for i in range(n_bars):
            t = b * bar
            ci = (b // 2) % 4  # two bars per chord
            for note in chords[ci]:
                tr.add(tone(midi_hz(note), bar, harmonics=(1, 0.4, 0.2)), t, 0.10 * energy,
                       pan=rng.uniform(-0.5, 0.5))
            tr.add(tone(midi_hz(roots[ci] - 12 + 12), bar, harmonics=(1, 0.3)), t, 0.30 * energy)
            if drums:
                for k in (0, 2):
                    tr.add(kick(), t + k * beat, 0.9 * energy)
                for s in (1, 3):
                    tr.add(snare(), t + s * beat, 0.35 * energy)
            for h in range(8):
                tr.add(hat(), t + h * beat / 2, 0.15 * energy, pan=0.3)
            if lead:
                for j in range(2):
                    note = melody[(b * 2 + j) % len(melody)]
                    tr.add(tone(midi_hz(note), 2 * beat * 0.95, harmonics=(1, 0.5, 0.25, 0.1),
                                vibrato=0.3), t + j * 2 * beat, 0.16 * energy)
            b += 1
    return tr.finish(total)


def song_b(bpm=145.0):
    beat = 60 / bpm
    bar = 4 * beat
    sections = [("intro", 8, False, 0.6), ("drop", 32, True, 1.0), ("break", 8, False, 0.6),
                ("drop", 32, True, 1.0), ("outro", 8, False, 0.5)]
    total = sum(s[1] for s in sections) * bar + 2
    tr = Track(total)
    prog = [(62, 65, 69), (58, 62, 65), (55, 58, 62), (57, 61, 64)]  # Dm Bb Gm A
    roots = [38, 34, 31, 33]
    b = 0
    for name, n_bars, full, energy in sections:
        for i in range(n_bars):
            t = b * bar
            ci = (b // 2) % 4
            # dark arp in 1/8ths
            for s in range(8):
                note = prog[ci][s % 3] + (12 if s % 4 == 3 else 0)
                tr.add(tone(midi_hz(note), beat / 2, decay=0.15, harmonics=(1, 0.5, 0.3)),
                       t + s * beat / 2, 0.12 * energy, pan=0.4 if s % 2 else -0.4)
            if full:
                tr.add(kick(), t, 1.0)
                tr.add(kick(), t + 2.5 * beat, 0.8)
                tr.add(bass808(midi_hz(roots[ci]), 2.4 * beat), t, 0.55)
                tr.add(bass808(midi_hz(roots[ci]), 1.4 * beat), t + 2.5 * beat, 0.45)
                tr.add(snare(), t + 2 * beat, 0.6)  # half-time clap on beat 3
                for h in range(8):
                    tr.add(hat(), t + h * beat / 2, 0.25, pan=0.2)
                if i % 4 == 3:  # 1/16 hat roll at the end of each 4-bar phrase
                    for h in range(8):
                        tr.add(hat(0.03), t + 3 * beat + h * beat / 8, 0.18, pan=0.2)
            else:
                tr.add(kick(), t, 0.5 * energy)
            b += 1
    return tr.finish(total)


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    a, b = song_a(), song_b()
    wav_a = os.path.join(out_dir, "song_a_moody.wav")
    sf.write(wav_a, a.T, SR)
    mp3_a = os.path.join(out_dir, "song_a_moody.mp3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_a, "-b:a", "256k", mp3_a], check=True)
    os.unlink(wav_a)
    sf.write(os.path.join(out_dir, "song_b_trap.wav"), b.T, SR)
    print(f"Wrote {mp3_a} ({a.shape[1] / SR:.0f}s) and song_b_trap.wav ({b.shape[1] / SR:.0f}s) to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples")
