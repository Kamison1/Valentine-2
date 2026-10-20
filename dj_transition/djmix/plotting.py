"""Matplotlib plot of the blend region with beat markers."""

from __future__ import annotations

import numpy as np

from .audio_io import min_max_envelope, to_mono


def plot_transition(session, settings, path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mix, plan, info, (stem_a, stem_b) = session.render_preview(settings, return_stems=True)
    w0, w1 = info["preview_window"]
    buckets = 3000
    markers = session.beat_markers(plan, w0, w1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    lanes = [
        ("Song A (outgoing)", stem_a, "#3b6fb6", markers["a"]),
        ("Song B (incoming)", stem_b, "#c0533a", markers["b"]),
        ("Merged", mix, "#444444", markers["a"] + markers["b"]),
    ]
    for ax, (title, sig, color, beats) in zip(axes, lanes):
        lo, hi = min_max_envelope(to_mono(sig), buckets)
        tb = np.linspace(w0, w1, len(lo))
        ax.fill_between(tb, lo, hi, color=color, linewidth=0)
        ax.axvspan(plan.t0, plan.t1, color="#f2c14e", alpha=0.25, label="blend")
        if plan.bass_swap:
            ax.axvline(plan.swap_time, color="#7a3fb0", linestyle="--", linewidth=1.2, label="bass swap")
        for bm in beats:
            ax.axvline(bm["t"], color="black", alpha=0.55 if bm["down"] else 0.12,
                       linewidth=1.0 if bm["down"] else 0.5)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(title, fontsize=9)
        ax.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time in merged track (s)")
    fig.suptitle(
        f"{settings.transition_bars}-bar transition - {info['tempo']}", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
