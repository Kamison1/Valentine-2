"""High-level mixing session shared by the CLI and the web app."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np

from . import DJMixError
from .alignment import (MixOutCandidate, grid_time, local_bpm, nearest_downbeat, plan_tempo,
                        suggest_mix_out_points)
from .analysis import TrackAnalysis, analyze_track, camelot_compatible, suggest_pitch_fix
from .audio_io import (SAMPLE_RATE, export_mp3, export_wav, load_audio, min_max_envelope,
                       prevent_clipping, to_mono)
from .mixing import TransitionPlan, render_mix, rubberband_available

TRANSITION_BAR_CHOICES = (4, 8, 16, 32)
PREVIEW_SECONDS = 30.0


@dataclass
class MixSettings:
    mix_out_time: float | None = None  # seconds in Song A; None = pick automatically
    b_offset_beats: int = 0  # which beat of B (relative to its first downbeat) lands on the mix-out
    transition_bars: int = 16
    stretch: str = "auto"  # "auto" | "a" (stretch A to B) | "b" (stretch B to A)
    multiplier: str = "auto"  # "auto" | "1" | "2" (A half-time) | "0.5" (A double-time)
    bass_swap: bool = True
    crossover_hz: float = 180.0
    vocal_cut: bool = True
    vocal_cut_db: float = -6.0
    pitch_semitones: int = 0
    pitch_target: str = "b"  # which track gets the pitch shift
    gain_match: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "MixSettings":
        s = cls()
        for f in fields(cls):
            if f.name in d and d[f.name] is not None:
                val = d[f.name]
                if f.name in ("bass_swap", "vocal_cut", "gain_match"):
                    val = val if isinstance(val, bool) else str(val).lower() in ("1", "true", "on", "yes")
                elif f.name in ("b_offset_beats", "transition_bars", "pitch_semitones"):
                    val = int(round(float(val)))
                elif f.name in ("mix_out_time", "crossover_hz", "vocal_cut_db"):
                    val = float(val)
                else:
                    val = str(val)
                setattr(s, f.name, val)
        s.validate()
        return s

    def validate(self) -> None:
        if self.transition_bars < 1 or self.transition_bars > 64:
            raise DJMixError("Transition length must be between 1 and 64 bars.")
        if not 40 <= self.crossover_hz <= 500:
            raise DJMixError("Bass-swap crossover must be between 40 and 500 Hz.")
        if abs(self.pitch_semitones) > 12:
            raise DJMixError("Pitch shift is limited to +/-12 semitones.")
        if self.pitch_target not in ("a", "b"):
            raise DJMixError("pitch_target must be 'a' or 'b'.")

    def to_dict(self) -> dict:
        return asdict(self)


class MixSession:
    """Two loaded + analysed songs, able to plan and render transitions."""

    def __init__(self, path_a: str, path_b: str, use_madmom: bool = True, log=print):
        self.log = log
        self.path_a, self.path_b = path_a, path_b
        self.sr = SAMPLE_RATE
        t = time.time()
        self.audio_a = load_audio(path_a, self.sr)
        self.audio_b = load_audio(path_b, self.sr)
        name_a, name_b = os.path.basename(path_a), os.path.basename(path_b)
        self.analysis_a = analyze_track(self.audio_a, self.sr, f"Song A ({name_a})", path_a, use_madmom)
        self.analysis_b = analyze_track(self.audio_b, self.sr, f"Song B ({name_b})", path_b, use_madmom)
        self.log(f"Analysed both tracks in {time.time() - t:.1f}s")
        for an in (self.analysis_a, self.analysis_b):
            self.log(f"  {an.name}: {an.bpm:.1f} BPM, key {an.key.long_name} ({an.camelot})")

    # ----------------------------------------------------------- analysis info

    @property
    def keys_compatible(self) -> bool:
        return camelot_compatible(self.analysis_a.camelot, self.analysis_b.camelot)

    def key_report(self, settings: MixSettings | None = None) -> dict:
        settings = settings or MixSettings()
        ka, kb = self.analysis_a.key, self.analysis_b.key
        if settings.pitch_target == "a":
            ka = ka.transposed(settings.pitch_semitones)
        else:
            kb = kb.transposed(settings.pitch_semitones)
        fix = suggest_pitch_fix(self.analysis_a.key, self.analysis_b.key)
        return {
            "key_a": ka.name, "camelot_a": ka.camelot,
            "key_b": kb.name, "camelot_b": kb.camelot,
            "compatible": camelot_compatible(ka.camelot, kb.camelot),
            "suggested_pitch_b": fix,
        }

    def suggestions(self, settings: MixSettings | None = None, top: int = 5) -> list[MixOutCandidate]:
        settings = settings or MixSettings()
        tp = plan_tempo(self.analysis_a.bpm, self.analysis_b.bpm, settings.stretch, settings.multiplier)
        blend_in_a = settings.transition_bars * tp.bar_seconds * tp.rate_a
        return suggest_mix_out_points(self.analysis_a, blend_in_a, top=top)

    def auto_settings(self, transition_bars: int = 16) -> MixSettings:
        s = MixSettings(transition_bars=transition_bars)
        s.mix_out_time = self.suggestions(s, top=1)[0].time
        return s

    # ---------------------------------------------------------------- planning

    def plan(self, settings: MixSettings) -> tuple[TransitionPlan, dict]:
        a, b = self.analysis_a, self.analysis_b
        if settings.mix_out_time is None:
            settings.mix_out_time = self.suggestions(settings, top=1)[0].time
        bar_a, mix_out = nearest_downbeat(a, settings.mix_out_time)

        b_down_idx = b.beat_index[b.downbeat_mask]
        start_idx = b_down_idx[0] + settings.b_offset_beats
        b_start = grid_time(b, start_idx)
        if b_start < 0:
            b_start = grid_time(b, start_idx + 4 * int(np.ceil(-b_start / b.bar_period)))
        if b_start >= b.duration - 5:
            raise DJMixError("B start offset is past the end of Song B.")

        tp = plan_tempo(local_bpm(a, mix_out), local_bpm(b, b_start), settings.stretch, settings.multiplier)
        length = settings.transition_bars * tp.bar_seconds
        t0 = mix_out / tp.rate_a
        warnings = list(a.warnings) + list(b.warnings) + list(tp.warnings)
        if not rubberband_available():
            warnings.append("Rubber Band not installed - using lower-quality librosa stretching.")

        if mix_out + length * tp.rate_a > a.duration:
            warnings.append("Song A ends before the blend finishes; the tail is padded with silence.")
        if b_start + length * tp.rate_b > b.duration:
            warnings.append("Song B is shorter than the blend.")

        gain_b = 1.0
        if settings.gain_match:
            gain_db = float(np.clip(a.rms_db - b.rms_db, -12.0, 6.0))
            gain_b = 10 ** (gain_db / 20)

        keys = self.key_report(settings)
        if not keys["compatible"]:
            fix = keys["suggested_pitch_b"]
            hint = f" Try pitch-shifting B by {fix:+d} semitones." if fix else ""
            warnings.append(
                f"Keys {keys['key_a']} ({keys['camelot_a']}) and {keys['key_b']} "
                f"({keys['camelot_b']}) are not harmonically compatible.{hint}"
            )

        plan = TransitionPlan(
            sr=self.sr, duration_a=a.duration, duration_b=b.duration,
            rate_a=tp.rate_a, rate_b=tp.rate_b,
            pitch_a=settings.pitch_semitones if settings.pitch_target == "a" else 0,
            pitch_b=settings.pitch_semitones if settings.pitch_target == "b" else 0,
            mix_out_a=mix_out, b_start=b_start, length=length,
            beat_sec=60.0 / tp.grid_bpm, bar_sec=tp.bar_seconds,
            swap_time=t0 + (settings.transition_bars // 2) * tp.bar_seconds,
            gain_b=gain_b, bass_swap=settings.bass_swap, crossover_hz=settings.crossover_hz,
            vocal_cut=settings.vocal_cut, vocal_cut_db=settings.vocal_cut_db,
        )
        info = {
            "tempo": tp.describe(),
            "bpm_a": round(tp.bpm_a, 2), "bpm_b": round(tp.bpm_b, 2),
            "played_bpm_a": round(tp.played_bpm_a, 2), "played_bpm_b": round(tp.played_bpm_b, 2),
            "multiplier": tp.multiplier, "stretched": tp.stretched,
            "stretch_pct": round(tp.change_pct, 2),
            "mix_out_time": round(mix_out, 3), "mix_out_bar": bar_a,
            "b_start": round(b_start, 3),
            "blend_start": round(plan.t0, 3), "blend_end": round(plan.t1, 3),
            "blend_seconds": round(length, 3), "swap_time": round(plan.swap_time, 3),
            "gain_b_db": round(20 * np.log10(gain_b), 2),
            "total_duration": round(plan.total_duration, 2),
            "keys": keys, "warnings": warnings,
        }
        return plan, info

    # --------------------------------------------------------------- rendering

    def preview_window(self, plan: TransitionPlan, seconds: float = PREVIEW_SECONDS) -> tuple[float, float]:
        """~30 s around the blend (longer if the blend itself is longer)."""
        if plan.length + 8 <= seconds:
            w0 = plan.t0 - (seconds - plan.length) / 2
            w1 = w0 + seconds
        else:
            w0, w1 = plan.t0 - 4, plan.t1 + 4
        shift = max(0.0, -w0)
        return w0 + shift, min(plan.total_duration, w1 + shift)

    def render_preview(self, settings: MixSettings, return_stems: bool = False):
        plan, info = self.plan(settings)
        w0, w1 = self.preview_window(plan)
        t = time.time()
        result = render_mix(self.audio_a, self.audio_b, plan, w0, w1, return_stems=return_stems)
        info["preview_window"] = [round(w0, 6), round(w1, 6)]
        info["render_seconds"] = round(time.time() - t, 2)
        if return_stems:
            mix, sa, sb = result
            return prevent_clipping(mix), plan, info, (sa, sb)
        return prevent_clipping(result), plan, info

    def render_full(self, settings: MixSettings):
        plan, info = self.plan(settings)
        t = time.time()
        mix = prevent_clipping(render_mix(self.audio_a, self.audio_b, plan))
        info["render_seconds"] = round(time.time() - t, 2)
        return mix, plan, info

    def export(self, mix: np.ndarray, out_path: str, mp3: bool = True) -> dict:
        base, ext = os.path.splitext(out_path)
        wav_path = base + ".wav" if ext.lower() != ".wav" else out_path
        paths = {"wav": export_wav(wav_path, mix, self.sr)}
        if mp3:
            try:
                paths["mp3"] = export_mp3(base + ".mp3", mix, self.sr)
            except DJMixError as exc:
                self.log(f"Warning: {exc}")
        return paths

    # --------------------------------------------------------- display helpers

    def beat_markers(self, plan: TransitionPlan, w0: float, w1: float) -> dict:
        """Beat positions of both tracks on the output timeline inside [w0, w1]."""
        out = {}
        for key, an, conv in (("a", self.analysis_a, plan.a_to_out), ("b", self.analysis_b, plan.b_to_out)):
            t = conv(an.beat_times)
            sel = (t >= w0) & (t <= w1)
            if key == "a":
                sel &= t <= plan.t1
            else:
                sel &= t >= plan.t0
            out[key] = [{"t": round(float(x), 4), "down": bool(d)}
                        for x, d in zip(t[sel], an.downbeat_mask[sel])]
        return out


def waveform_peaks(y: np.ndarray, buckets: int = 1200) -> list[list[float]]:
    """Min/max pairs per bucket of the mono signal, for drawing waveforms."""
    mono = to_mono(y)
    if len(mono) == 0:
        return []
    mins, maxs = min_max_envelope(mono, buckets)
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(mins, maxs)]
