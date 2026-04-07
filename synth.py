#!/usr/bin/env python3
"""
synth.py

Granular concatenative speech synthesiser.

Given a sequence of ARPAbet phonemes (or a word that is looked up via the
TextGrid), this script:

  1. Loads the grain index produced by extract_grains.py.
  2. For each phoneme, selects a grain from the corpus (random by default,
     or nearest-duration to a target).
  3. Concatenates grains using overlap-add (OLA) crossfading — the same
     mechanism UTAU uses between samples.
  4. Writes the result to an output wav.

Usage examples
--------------
  # synthesise from an explicit phone sequence
  python synth.py --phones "HH EH L OW"

  # look up a word from the TextGrid and resynthesize it
  python synth.py --word "nasdaq"

  # target specific grain durations (seconds, space-separated, matches --phones)
  python synth.py --phones "HH EH L OW" --durations "0.05 0.10 0.08 0.12"

  # pick the longest available grain for each phoneme instead of random
  python synth.py --phones "HH EH L OW" --strategy longest
"""

import argparse
import json
import os
import random
import wave
import numpy as np
import tgt
import aligned_textgrid as atg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAIN_INDEX = "grains/index.json"
TG_PATH     = "aligned/bonfire.TextGrid"
OUT_WAV     = "out_synth.wav"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as wf:
        sr        = wf.getframerate()
        n_frames  = wf.getnframes()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw       = wf.readframes(n_frames)
    dtype   = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    peak    = float(2 ** (8 * sampwidth - 1))
    samples /= peak
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)
    return samples, sr


def write_wav_mono(path: str, samples: np.ndarray, sr: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm     = (clipped * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def load_index(path: str) -> dict[str, list[dict]]:
    """Return {phoneme: [grain_meta, …]} from index.json."""
    with open(path) as f:
        entries = json.load(f)
    corpus: dict[str, list[dict]] = {}
    for entry in entries:
        corpus.setdefault(entry["phoneme"], []).append(entry)
    return corpus


# ---------------------------------------------------------------------------
# Grain selection strategies
# ---------------------------------------------------------------------------

def select_grain(
    corpus: dict[str, list[dict]],
    phoneme: str,
    strategy: str = "random",
    target_dur: float | None = None,
) -> dict | None:
    """Pick one grain metadata dict for *phoneme* according to *strategy*."""
    candidates = corpus.get(phoneme)
    if not candidates:
        print(f"  [warn] no grain found for phoneme '{phoneme}' — skipping")
        return None

    if strategy == "random":
        return random.choice(candidates)
    elif strategy == "longest":
        return max(candidates, key=lambda g: g["duration"])
    elif strategy == "shortest":
        return min(candidates, key=lambda g: g["duration"])
    elif strategy == "nearest" and target_dur is not None:
        return min(candidates, key=lambda g: abs(g["duration"] - target_dur))
    else:
        return random.choice(candidates)


# ---------------------------------------------------------------------------
# Overlap-add concatenation
# ---------------------------------------------------------------------------

def ola_concat(grains: list[np.ndarray], overlap_samples: int) -> np.ndarray:
    """
    Concatenate grains with overlap-add crossfading.

    Each grain is assumed to have Hann-faded edges (written by extract_grains).
    The overlap region of consecutive grains is summed, which gives a smooth
    join without clicks.
    """
    if not grains:
        return np.array([], dtype=np.float64)
    if len(grains) == 1:
        return grains[0]

    # total length: sum of grain lengths minus the overlaps between each pair
    total = sum(len(g) for g in grains) - overlap_samples * (len(grains) - 1)
    total = max(total, sum(len(g) for g in grains))  # safety floor
    out   = np.zeros(total, dtype=np.float64)

    pos = 0
    for i, grain in enumerate(grains):
        end = pos + len(grain)
        if end > len(out):
            # expand buffer if needed (can happen with variable grain lengths)
            out = np.concatenate([out, np.zeros(end - len(out))])
        out[pos:end] += grain
        pos += len(grain) - overlap_samples

    return out[:pos + overlap_samples]


# ---------------------------------------------------------------------------
# TextGrid word → phone lookup
# ---------------------------------------------------------------------------

def phones_for_word(word: str) -> list[str]:
    """Return the ARPAbet phone sequence for *word* from the TextGrid."""
    tg = atg.AlignedTextGrid(
        textgrid_path=TG_PATH,
        entry_classes=[atg.Word, atg.Phone],
    )
    group = tg[0]
    words_tier = group[0]

    for w in words_tier:
        if w.label.lower() == word.lower():
            return [p.label for p in w.contains if p.label.strip()]

    raise ValueError(f"Word '{word}' not found in TextGrid.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Granular speech synthesiser")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--phones",    type=str, help="Space-separated ARPAbet phones, e.g. 'HH EH L OW'")
    src.add_argument("--word",      type=str, help="Word to look up in the TextGrid")
    parser.add_argument("--durations", type=str, default=None,
                        help="Target grain durations in seconds, space-separated (used with --phones and --strategy nearest)")
    parser.add_argument("--strategy", choices=["random", "longest", "shortest", "nearest"],
                        default="random", help="Grain selection strategy (default: random)")
    parser.add_argument("--overlap", type=float, default=0.015,
                        help="Crossfade half-width in seconds (default: 0.015)")
    parser.add_argument("--out", type=str, default=OUT_WAV,
                        help=f"Output wav path (default: {OUT_WAV})")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # -- load corpus ---------------------------------------------------------
    print(f"Loading grain index from {GRAIN_INDEX} …")
    corpus = load_index(GRAIN_INDEX)
    print(f"  {sum(len(v) for v in corpus.values())} grains  |  {len(corpus)} phoneme classes")

    # -- resolve phone sequence ----------------------------------------------
    if args.word:
        phones = phones_for_word(args.word)
        print(f"  '{args.word}' → {phones}")
    else:
        phones = args.phones.split()
        print(f"  phones: {phones}")

    # -- resolve target durations (for 'nearest' strategy) ------------------
    target_durs: list[float | None] = [None] * len(phones)
    if args.durations:
        raw = [float(x) for x in args.durations.split()]
        target_durs = (raw + [None] * len(phones))[:len(phones)]

    # -- select & load grains ------------------------------------------------
    # peek at sr from first available grain
    first_path = next(
        (g["path"] for gs in corpus.values() for g in gs), None
    )
    if first_path is None:
        raise RuntimeError("Grain index is empty — run extract_grains.py first.")
    _, sr = read_wav_mono(first_path)
    overlap_samp = max(1, int(args.overlap * sr))

    grain_arrays: list[np.ndarray] = []
    for phone, target in zip(phones, target_durs):
        meta = select_grain(corpus, phone, strategy=args.strategy, target_dur=target)
        if meta is None:
            continue
        audio, _ = read_wav_mono(meta["path"])
        grain_arrays.append(audio)
        print(f"  {phone:6s}  {meta['path']}  ({meta['duration']:.3f}s)")

    if not grain_arrays:
        print("No grains loaded — nothing to synthesise.")
        return

    # -- overlap-add concat --------------------------------------------------
    result = ola_concat(grain_arrays, overlap_samp)

    # -- write output --------------------------------------------------------
    write_wav_mono(args.out, result, sr)
    print(f"\nWrote {len(result)/sr:.3f}s  →  {args.out}")


if __name__ == "__main__":
    main()
