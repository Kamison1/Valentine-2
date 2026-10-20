"""Command-line interface: ``djmix merge`` and ``djmix web``."""

from __future__ import annotations

import argparse
import os
import sys

from . import DJMixError
from .alignment import MULTIPLIERS, STRETCH_MODES, suggest_mix_out_points
from .analysis import suggest_pitch_fix


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def _print_analysis(session) -> None:
    for an in (session.analysis_a, session.analysis_b):
        print(f"\n{an.name}")
        print(f"  Duration : {_fmt_time(an.duration)}")
        print(f"  BPM      : {an.bpm:.2f}  (beat jitter {an.tempo_stability:.0f} ms, "
              f"downbeats via {an.downbeat_method})")
        print(f"  Key      : {an.key.long_name}  ->  Camelot {an.camelot}"
              f"  (confidence {an.key_confidence:.2f})")
        for bars in (16, 32):
            cands = suggest_mix_out_points(an, bars * an.bar_period, top=3)
            print(f"  Suggested mix-out points for a {bars}-bar blend:")
            for c in cands:
                print(f"    {_fmt_time(c.time)} ({c.time:7.2f}s)  {c.reason}")
        for w in an.warnings:
            print(f"  ! {w}")

    a, b = session.analysis_a, session.analysis_b
    print()
    if session.keys_compatible:
        print(f"Keys are harmonically compatible ({a.camelot} -> {b.camelot}).")
    else:
        fix = suggest_pitch_fix(a.key, b.key)
        hint = f" Suggested fix: --pitch-shift {fix:+d}" if fix else ""
        print(f"WARNING: keys {a.camelot} and {b.camelot} are not harmonically compatible.{hint}")


def _settings_from_args(args, session):
    from .engine import MixSettings

    s = MixSettings(
        mix_out_time=args.mix_out_seconds, b_offset_beats=args.b_offset_beats,
        transition_bars=args.transition_bars, stretch=args.stretch, multiplier=args.multiplier,
        bass_swap=not args.no_bass_swap, crossover_hz=args.crossover_hz,
        vocal_cut=not args.no_vocal_cut, vocal_cut_db=args.vocal_cut_db,
        pitch_semitones=args.pitch_shift, pitch_target=args.pitch_target,
        gain_match=not args.no_gain_match,
    )
    s.validate()
    if s.mix_out_time is None:
        best = session.suggestions(s, top=1)[0]
        print(f"Auto mix-out point: {_fmt_time(best.time)} - {best.reason}")
        s.mix_out_time = best.time
    return s


def cmd_merge(args) -> int:
    from .engine import MixSession
    from .plotting import plot_transition

    session = MixSession(args.song_a, args.song_b, use_madmom=not args.no_madmom)
    if args.analyze_only:
        _print_analysis(session)
        return 0

    settings = _settings_from_args(args, session)
    base = os.path.splitext(args.output)[0]
    if args.preview_only:
        mix, plan, info = session.render_preview(settings)
        out_name = base + "_preview"
    else:
        mix, plan, info = session.render_full(settings)
        out_name = base

    print(f"\nTempo    : {info['tempo']}")
    print(f"Keys     : {info['keys']['key_a']} ({info['keys']['camelot_a']}) -> "
          f"{info['keys']['key_b']} ({info['keys']['camelot_b']})")
    print(f"Mix-out  : {_fmt_time(info['mix_out_time'])} in Song A (bar {info['mix_out_bar'] + 1}); "
          f"Song B enters at {info['b_start']:.2f}s")
    print(f"Blend    : {_fmt_time(info['blend_start'])} - {_fmt_time(info['blend_end'])} in the merged "
          f"track ({settings.transition_bars} bars, {info['blend_seconds']:.1f}s)")
    for w in info["warnings"]:
        print(f"WARNING: {w}")

    paths = session.export(mix, out_name + ".wav", mp3=not args.no_mp3)
    for kind, p in paths.items():
        print(f"Wrote {kind.upper()}: {p}")
    if not args.no_plot:
        png = args.plot or base + "_transition.png"
        plot_transition(session, settings, png)
        print(f"Wrote plot: {png}")
    return 0


def cmd_web(args) -> int:
    from .web.app import create_app

    app = create_app(args.workdir)
    print(f"DJ transition tool running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="djmix", description="DJ-style transitions between two songs.")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("merge", help="Analyse two songs and render the merged track.")
    m.add_argument("song_a", help="Outgoing song (mp3/wav/flac)")
    m.add_argument("song_b", help="Incoming song (mp3/wav/flac)")
    m.add_argument("--output", "-o", default="merged.wav",
                   help="Output path; .wav and .mp3 are both written (default: merged.wav)")
    m.add_argument("--transition-bars", type=int, default=16, help="Blend length in bars (default 16)")
    m.add_argument("--mix-out-seconds", type=float, default=None,
                   help="Where in Song A to start the blend (snapped to the nearest bar). Default: auto")
    m.add_argument("--analyze-only", action="store_true",
                   help="Print BPM, key, Camelot code and suggested mix-out points, don't render")
    m.add_argument("--b-offset-beats", type=int, default=0,
                   help="Shift which beat of Song B lands on the mix-out downbeat")
    m.add_argument("--stretch", choices=STRETCH_MODES, default="auto",
                   help="Which track to time-stretch: auto (smallest change), a, or b")
    m.add_argument("--multiplier", choices=MULTIPLIERS, default="auto",
                   help="Tempo relation B = k x A: auto, 1, 2 (A half-time), 0.5 (A double-time)")
    m.add_argument("--no-bass-swap", action="store_true", help="Disable the bass swap")
    m.add_argument("--crossover-hz", type=float, default=180.0, help="Bass swap crossover (default 180)")
    m.add_argument("--no-vocal-cut", action="store_true", help="Disable the vocal-range EQ cut on A")
    m.add_argument("--vocal-cut-db", type=float, default=-6.0, help="Depth of the vocal cut (default -6)")
    m.add_argument("--pitch-shift", type=int, default=0, help="Pitch shift in semitones (default 0)")
    m.add_argument("--pitch-target", choices=("a", "b"), default="b", help="Track to pitch-shift")
    m.add_argument("--no-gain-match", action="store_true", help="Don't match B's loudness to A")
    m.add_argument("--preview-only", action="store_true", help="Render only ~30s around the transition")
    m.add_argument("--plot", default=None, help="PNG path for the blend plot (default <output>_transition.png)")
    m.add_argument("--no-plot", action="store_true", help="Skip the waveform plot")
    m.add_argument("--no-mp3", action="store_true", help="Only write WAV")
    m.add_argument("--no-madmom", action="store_true", help="Don't use madmom even if installed")
    m.set_defaults(func=cmd_merge)

    w = sub.add_parser("web", help="Start the interactive web app.")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=5000)
    w.add_argument("--workdir", default=None, help="Where uploads/renders are stored (default: temp dir)")
    w.set_defaults(func=cmd_web)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DJMixError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
