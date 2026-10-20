# djmix: DJ transition tool

`djmix` blends two songs the way a DJ would and exports one merged audio file. It
beat-matches the tracks, lines up their downbeats on a phrase boundary, crossfades,
swaps the basses and cuts A's vocal range during the blend. **Auto-merge** picks
the settings for you. After that you can tweak the transition in a small local web
app, where each change re-renders only a ~30 s preview. There's also a CLI that uses
the same engine.

```
Song A ──────────────[ blend: N bars ]
                     [ B (highs only) | bass swap | B full ]────────── Song B
```

## Install

Python 3.10+ is required. You also need two command-line tools:

| Tool | Why | macOS | Ubuntu/Debian | Windows |
|---|---|---|---|---|
| **ffmpeg** | MP3 decode/encode | `brew install ffmpeg` | `sudo apt install ffmpeg` | `winget install ffmpeg` |
| **Rubber Band** | high-quality time-stretch / pitch-shift | `brew install rubberband` | `sudo apt install rubberband-cli` | download the CLI from <https://breakfastquay.com/rubberband/>, put `rubberband.exe` on `PATH` |

Then:

```bash
cd dj_transition
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # or: pip install -e .[dev]
```

Optional extras:
- **madmom** (`pip install madmom`) gives better downbeat detection. It doesn't build on every Python version. Without it, a built-in heuristic is used (kick/bass energy plus chord changes).
- Without Rubber Band, stretching falls back to librosa's phase vocoder. It works, but sounds worse, and you'll see a warning.

## Quick start with demo audio

```bash
python scripts/make_samples.py samples          # synthetic 77 BPM "moody" song + 145 BPM trap song
python -m djmix web                             # open http://127.0.0.1:5000
```

## Web app

```bash
python -m djmix web [--port 5000] [--workdir ./renders]
```

1. Choose Song A (outgoing) and Song B (incoming), then click **Auto-merge**. Both songs are analysed, and the first full version is rendered with the best automatic settings.
2. You'll see BPM, key and Camelot code for both songs, a compatibility badge, and any warnings (key clash, stretch above 8%, unsteady tempo, ...).
3. Full-track waveforms of A and B show bar ticks and the blend region. Click A's waveform to move the mix-out point.
4. The transition preview shows A, B and the merged signal on one timeline, with beat markers (downbeats brighter), the blend in yellow and the bass swap in purple. Click it to play from that point.
5. Controls. Each change re-renders the preview in about 1–2 s:
   - **Mix-out point** slider (snaps to bars), with buttons for the suggested phrase boundaries
   - **Song B start offset** in beats from B's first downbeat
   - **Transition length**: 4 / 8 / 16 / 32 bars
   - **Tempo match**: auto, stretch A to B, or stretch B to A, plus a **half-/double-time** selector
   - **Bass swap** on/off and its crossover frequency
   - **Vocal-clash EQ cut** on/off and its depth
   - **Pitch shift** in semitones, on A or B. "Apply suggested" picks the smallest shift that fixes a key clash.
   - **Loudness match** of B to A
6. **Render full track** exports the final WAV and MP3 (download links next to the player).

## CLI

```bash
# Analyse only: BPM, key, Camelot code, suggested mix-out points (16 and 32-bar blends)
python -m djmix merge songA.mp3 songB.wav --analyze-only

# Fully automatic merge -> merged.wav, merged.mp3 and merged_transition.png
python -m djmix merge songA.mp3 songB.wav --output merged.wav

# Manual choices
python -m djmix merge songA.mp3 songB.flac -o mix.wav \
    --transition-bars 32 --mix-out-seconds 182.5 --b-offset-beats 4 \
    --stretch b --crossover-hz 150 --pitch-shift -1

# Quick 30 s render around the transition only
python -m djmix merge songA.mp3 songB.wav -o test.wav --preview-only
```

With `pip install -e .` the `djmix` command is also available directly.

Main flags: `--transition-bars` (default 16), `--mix-out-seconds`, `--output`,
`--analyze-only`, `--b-offset-beats`, `--stretch {auto,a,b}`,
`--multiplier {auto,1,2,0.5}`, `--no-bass-swap`, `--crossover-hz`, `--no-vocal-cut`,
`--vocal-cut-db`, `--pitch-shift`, `--pitch-target {a,b}`, `--no-gain-match`,
`--plot PATH`, `--no-plot`, `--no-mp3`, `--preview-only`. Run `python -m djmix merge -h`
for the full list.

## How it works

| Step | Module | What happens |
|---|---|---|
| Load | `audio_io.py` | librosa decode, resampled to 44.1 kHz stereo (mono is duplicated). Clear errors for unsupported, missing, too-short, silent or undecodable files. |
| Beats and BPM | `analysis.py` | librosa beat tracking on a 22.05 kHz mono copy. BPM comes from a least-squares fit of the beat grid, which is much more precise than the median interval. The grid is extended through intros and outros the tracker trimmed. A pulse-clarity check rejects beatless material. |
| Downbeats | `analysis.py` | madmom RNN if it's installed. Otherwise, the bar phase where low-band (kick/808) onsets and chord changes are strongest. |
| Key | `analysis.py` | CQT chroma correlated with Krumhansl–Kessler profiles, giving 24 keys mapped to Camelot codes. Compatible means same code, relative major/minor, or ±1 on the wheel. |
| Tempo plan | `alignment.py` | Picks ×1, ×2 or ×½ (e.g. 77 vs 145 → A is half-time) and stretches whichever track needs the smaller change. Warns above 8%. Uses the local tempo around the transition points. |
| Mix-out | `alignment.py` | Ranks bar lines in the second half of A. A phrase start (multiple of 8/16 bars) scores well, as does a blend that ends exactly where a section of A ends (energy change in dB, or the end of the music). |
| Blend | `mixing.py` | Equal-power crossfade over N bars. Bass swap: complementary zero-phase low/high split at the crossover, where A's lows leave in the beat before the swap point (halfway through the blend) and B's lows arrive in the beat after, so the basses never overlap. A's 300 Hz–3 kHz band is cut (default −6 dB) during the overlap. |
| Render | `mixing.py` | Rubber Band (via pyrubberband) stretches and pitch-shifts in a single pass, and only on the part of each song that's needed. That's why previews are fast. The full render is A up to the blend, then the blend, then the rest of B, with clipping protection. |

## Tests

```bash
python -m pytest -q
```

The tests generate short tone tracks (kick pulses plus a chord) at known tempos and keys.
They check BPM/key/downbeat detection, Camelot logic, half/double-time planning, the
stretch warning, full and preview rendering and export, the bass-swap behaviour, and
the error messages for bad files and beatless audio.

## Limitations

- Beat tracking is the weak point for rubato or live material. An unsteady tempo produces a warning, and the alignment uses a local beat-grid fit around the transition.
- Half-time ambiguity is common: a 77 BPM song may be detected as 154. The tempo plan treats both as compatible either way, but "bars" for that song are then counted at the detected tempo.
- Key detection gives one global key per song. Relative major/minor confusion (e.g. Am vs C) doesn't matter for Camelot compatibility, because both share the same number.
- The preview and the full render peak-normalise independently, so their levels can differ slightly.
