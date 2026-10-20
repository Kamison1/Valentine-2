import os

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from conftest import SR, make_track
from djmix import AudioLoadError, BeatDetectionError, DJMixError
from djmix.alignment import plan_tempo
from djmix.analysis import Key, analyze_track, camelot_compatible, suggest_pitch_fix
from djmix.audio_io import load_audio
from djmix.engine import MixSettings
from djmix.mixing import render_mix


# ------------------------------------------------------------------ keys

def test_camelot_codes():
    assert Key(0, False).camelot == "8B"   # C major
    assert Key(9, True).camelot == "8A"    # A minor
    assert Key(7, False).camelot == "9B"   # G major
    assert Key(6, True).camelot == "11A"   # F# minor
    assert Key(11, False).camelot == "1B"  # B major


def test_camelot_compatibility():
    assert camelot_compatible("8A", "8A")
    assert camelot_compatible("8A", "8B")    # relative major/minor
    assert camelot_compatible("12A", "1A")   # wraps around the wheel
    assert not camelot_compatible("8A", "10A")
    assert not camelot_compatible("8A", "9B")
    # C major (8B) vs D major (10B): shifting B down 2 semitones lands on C major.
    assert suggest_pitch_fix(Key(0, False), Key(2, False)) == -2
    assert suggest_pitch_fix(Key(0, False), Key(7, False)) == 0      # already compatible
    # D minor (7A) only fits C major via A minor, 5+ semitones away: nothing within +/-3.
    assert suggest_pitch_fix(Key(0, False), Key(2, True)) is None


# ----------------------------------------------------------------- tempo

def test_half_double_time_is_compatible():
    plan = plan_tempo(77.0, 145.0)
    assert plan.multiplier == 2.0
    assert plan.stretched == "a"                       # slowing A is the smaller change
    assert plan.played_bpm_a * 2 == pytest.approx(145.0, rel=1e-6)
    assert any("half/double" in w for w in plan.warnings)

    assert plan_tempo(150.0, 75.0).multiplier == 0.5
    forced = plan_tempo(77.0, 145.0, stretch="b")
    assert forced.rate_a == 1.0 and forced.played_bpm_b == pytest.approx(154.0)


def test_large_stretch_warns():
    plan = plan_tempo(120.0, 140.0)
    assert abs(plan.change_pct) > 8
    assert any("8%" in w for w in plan.warnings)
    assert not plan_tempo(120.0, 124.0).warnings


# -------------------------------------------------------------- analysis

def test_bpm_and_key_detection(session):
    a, b = session.analysis_a, session.analysis_b
    assert a.bpm == pytest.approx(120.0, rel=0.015)
    assert b.bpm == pytest.approx(124.0, rel=0.015)
    assert a.camelot in ("8A", "8B")   # A minor triad (relative C major also acceptable)
    assert b.camelot in ("9A", "9B")   # E minor triad
    # Accented beats are every 2.0 s at 120 BPM; downbeats must land on them.
    phase = np.mod(a.downbeat_times + 0.05, 2.0)
    assert np.all(phase < 0.1), "downbeats should be on the accented beats"


def test_suggested_mix_out_is_on_a_bar(session):
    cands = session.suggestions(MixSettings(transition_bars=4))
    assert cands
    downs = session.analysis_a.downbeat_times
    for c in cands:
        assert np.min(np.abs(downs - c.time)) < 1e-6
        assert c.time > session.analysis_a.duration * 0.4


# ---------------------------------------------------------------- render

def test_full_render_and_export(session, tmp_path):
    settings = session.auto_settings(transition_bars=4)
    mix, plan, info = session.render_full(settings)
    assert mix.shape[0] == 2
    assert mix.shape[1] / SR == pytest.approx(plan.total_duration, abs=0.05)
    assert np.all(np.isfinite(mix)) and np.max(np.abs(mix)) <= 1.0
    # 120 -> 124 BPM is a ~3% stretch on one of the tracks
    assert abs(info["stretch_pct"]) == pytest.approx(3.3, abs=0.6)

    paths = session.export(mix, str(tmp_path / "out.wav"))
    assert os.path.getsize(paths["wav"]) > 0
    if "mp3" in paths:
        assert os.path.getsize(paths["mp3"]) > 0

    preview, _, pinfo = session.render_preview(settings)
    w0, w1 = pinfo["preview_window"]
    assert preview.shape[1] / SR == pytest.approx(w1 - w0, abs=0.01)
    assert w0 <= plan.t0 and w1 >= plan.t1


def test_bass_swap_removes_a_low_end(session):
    settings = session.auto_settings(transition_bars=8)
    plan, _ = session.plan(settings)
    _, stem_a, stem_b = render_mix(session.audio_a, session.audio_b, plan,
                                   plan.t0 - 2, plan.t1, return_stems=True)
    lp = butter(4, 120, btype="lowpass", fs=SR, output="sos")
    low_a = sosfiltfilt(lp, stem_a.mean(0))
    low_b = sosfiltfilt(lp, stem_b.mean(0))

    def rms(x, t0, t1):
        i0, i1 = int((t0 - (plan.t0 - 2)) * SR), int((t1 - (plan.t0 - 2)) * SR)
        return np.sqrt(np.mean(x[i0:i1] ** 2))

    before = rms(low_a, plan.t0 - 2, plan.t0)
    after = rms(low_a, plan.swap_time + 0.05, plan.t1)
    assert after < 0.05 * before                                  # A's bass is gone after the swap
    assert rms(low_b, plan.t0, plan.swap_time - 0.05) < 1e-3      # B's bass waits for the swap
    assert rms(low_b, plan.swap_time + plan.beat_sec, plan.t1) > 0.01


# ---------------------------------------------------------------- errors

def test_unsupported_and_bad_files(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("not audio")
    with pytest.raises(AudioLoadError, match="Unsupported file type"):
        load_audio(str(txt))

    with pytest.raises(AudioLoadError, match="not found"):
        load_audio(str(tmp_path / "missing.wav"))

    short = tmp_path / "short.wav"
    sf.write(short, make_track(120, 3).T, SR)
    with pytest.raises(AudioLoadError, match="at least"):
        load_audio(str(short))

    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"\x00" * 2048)
    with pytest.raises(AudioLoadError, match="Could not decode"):
        load_audio(str(broken))


def test_beat_detection_failure_is_reported():
    t = np.arange(int(15 * SR)) / SR
    hum = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    with pytest.raises(BeatDetectionError):
        analyze_track(np.stack([hum, hum]), SR, "drone", use_madmom=False)


def test_settings_validation():
    with pytest.raises(DJMixError):
        MixSettings.from_dict({"crossover_hz": 5000})
    s = MixSettings.from_dict({"bass_swap": "false", "transition_bars": "8", "pitch_semitones": "-2"})
    assert s.bass_swap is False and s.transition_bars == 8 and s.pitch_semitones == -2


def test_cli_analyze_only(tone_files, capsys):
    from djmix.cli import main

    assert main(["merge", *tone_files, "--analyze-only", "--no-madmom"]) == 0
    out = capsys.readouterr().out
    assert "BPM" in out and "Camelot" in out and "Suggested mix-out" in out
