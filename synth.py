#!/usr/bin/env python3
"""
synth.py

Granular concatenative speech synthesiser.

Given orthographic text (default), an explicit ARPAbet phone sequence, or a
word from the TextGrid, this script:

  1. Resolves the input to a flat ARPAbet phone sequence (via g2p.py for text).
  2. Loads the grain index produced by extract_grains.py.
  3. For each phoneme, selects a grain from the corpus.
  4. Concatenates grains using overlap-add (OLA) crossfading.
  5. Writes the result to an output wav.

Usage examples
--------------
  # default: orthographic text → G2P → synthesis
  python synth.py "hello world"

  # H&B-style cost-minimising grain selection
  python synth.py "the quick brown fox" --strategy cost

  # bypass G2P with an explicit ARPAbet sequence
  python synth.py --phones "HH EH L OW"

  # look up a word from the TextGrid and resynthesize it
  python synth.py --word "nasdaq"

  # target specific grain durations (used with --strategy nearest)
  python synth.py --phones "HH EH L OW" --durations "0.05 0.10 0.08 0.12"
"""

import argparse
import json
import os
import random
import wave
import numpy as np
import aligned_textgrid as atg
from g2p import text_to_phones

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAIN_INDEX = "grains/index.json"
TG_PATH     = "aligned/bonfire.TextGrid"
OUT_WAV     = "out_synth.wav"

# ---------------------------------------------------------------------------
# H&B-style cost weights
# ---------------------------------------------------------------------------
W_TARGET   = 1.0   # weight for phonetic context mismatch  (range 0–1)
W_JOIN     = 1.0   # weight for spectral join cost
MFCC_SCALE = 20.0  # normalising divisor for MFCC Euclidean distance
               #   ~5–15 = within-class variation; ~20–40 = cross-class

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


def load_index(path: str) -> tuple[dict, dict]:
    """
    Return (mono_corpus, context_index) from index.json.

    mono_corpus:   {phoneme: [grain_meta, …]}
    context_index: {(prev_phone, phoneme, next_phone): [grain_meta, …]}

    MFCC vectors stored as JSON lists are converted to numpy arrays here so
    distance computation in select_grain is fast.
    """
    with open(path) as f:
        entries = json.load(f)

    mono:    dict[str, list[dict]] = {}
    context: dict[tuple, list[dict]] = {}

    for entry in entries:
        for key in ("mfcc_entry", "mfcc_exit"):
            if key in entry:
                entry[key] = np.asarray(entry[key])

        ph   = entry["phoneme"]
        prev = entry.get("prev_phone")
        nxt  = entry.get("next_phone")

        mono.setdefault(ph, []).append(entry)
        context.setdefault((prev, ph, nxt), []).append(entry)

    return mono, context


# ---------------------------------------------------------------------------
# Grain selection strategies
# ---------------------------------------------------------------------------

def _select_by_cost(
    candidates: list[dict],
    prev_phone: str | None,
    next_phone: str | None,
    prev_mfcc_exit: np.ndarray | None,
    target_dur: float | None,
) -> dict:
    """
    Simplified Hunt & Black (1996) cost: target_cost + join_cost.

    target_cost  — phonetic context mismatch (0 = perfect triphone,
                   0.5 = one neighbour wrong, 1.0 = neither matches)
                   plus an optional duration penalty.
    join_cost    — normalised MFCC Euclidean distance between the exit
                   frame of the previous grain and the entry frame of
                   this candidate (0 when no previous grain exists).
    """
    def cost(g: dict) -> float:
        # target cost: context match (0–1)
        ctx = sum([g.get("prev_phone") == prev_phone,
                   g.get("next_phone") == next_phone])
        t_cost = (2 - ctx) / 2

        # optional duration penalty, normalised to ~0–1
        if target_dur is not None:
            t_cost += 0.5 * abs(g["duration"] - target_dur) / max(target_dur, 1e-6)

        # join cost: spectral distance at concatenation point
        if prev_mfcc_exit is not None and "mfcc_entry" in g:
            dist   = np.linalg.norm(prev_mfcc_exit - g["mfcc_entry"])
            j_cost = dist / MFCC_SCALE
        else:
            j_cost = 0.0

        return W_TARGET * t_cost + W_JOIN * j_cost

    return min(candidates, key=cost)


def select_grain(
    mono: dict[str, list[dict]],
    phoneme: str,
    strategy: str = "random",
    target_dur: float | None = None,
    prev_phone: str | None = None,
    next_phone: str | None = None,
    prev_mfcc_exit: np.ndarray | None = None,
) -> dict | None:
    """Pick one grain metadata dict for *phoneme* according to *strategy*."""
    candidates = mono.get(phoneme)
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
    elif strategy == "cost":
        return _select_by_cost(candidates, prev_phone, next_phone,
                               prev_mfcc_exit, target_dur)
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
    parser.add_argument("text", nargs="?", default=None,
                        help="Orthographic text to synthesise (default input mode)")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--phones", type=str,
                     help="ARPAbet phone sequence, e.g. 'HH EH L OW' (overrides positional text)")
    src.add_argument("--word",   type=str,
                     help="Single word to look up in the TextGrid (overrides positional text)")
    parser.add_argument("--durations", type=str, default=None,
                        help="Target grain durations in seconds, space-separated (used with --strategy nearest/cost)")
    parser.add_argument("--strategy", choices=["random", "longest", "shortest", "nearest", "cost"],
                        default="random", help="Grain selection strategy (default: random)")
    parser.add_argument("--overlap", type=float, default=0.015,
                        help="Crossfade half-width in seconds (default: 0.015)")
    parser.add_argument("--out", type=str, default=OUT_WAV,
                        help=f"Output wav path (default: {OUT_WAV})")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not any([args.text, args.phones, args.word]):
        parser.error("provide text to synthesise, --phones, or --word")

    if args.seed is not None:
        random.seed(args.seed)

    # -- load corpus ---------------------------------------------------------
    print(f"Loading grain index from {GRAIN_INDEX} …")
    mono, context_index = load_index(GRAIN_INDEX)
    print(f"  {sum(len(v) for v in mono.values())} grains  |  {len(mono)} phoneme classes")

    # -- resolve phone sequence ----------------------------------------------
    if args.phones:
        phones = args.phones.split()
        print(f"  phones: {phones}")
    elif args.word:
        phones = phones_for_word(args.word)
        print(f"  '{args.word}' → {phones}")
    else:
        print(f"  text: '{args.text}'")
        phones = text_to_phones(args.text)
        print(f"  → {phones}")

    # -- resolve target durations (for 'nearest'/'cost' strategies) ---------
    target_durs: list[float | None] = [None] * len(phones)
    if args.durations:
        raw = [float(x) for x in args.durations.split()]
        target_durs = (raw + [None] * len(phones))[:len(phones)]

    # -- select & load grains ------------------------------------------------
    first_path = next(
        (g["path"] for gs in mono.values() for g in gs), None
    )
    if first_path is None:
        raise RuntimeError("Grain index is empty — run extract_grains.py first.")
    _, sr = read_wav_mono(first_path)
    overlap_samp = max(1, int(args.overlap * sr))

    grain_arrays: list[np.ndarray] = []
    prev_mfcc_exit: np.ndarray | None = None

    for i, (phone, target) in enumerate(zip(phones, target_durs)):
        prev_phone = phones[i - 1] if i > 0            else None
        next_phone = phones[i + 1] if i < len(phones) - 1 else None

        meta = select_grain(
            mono, phone,
            strategy       = args.strategy,
            target_dur     = target,
            prev_phone     = prev_phone,
            next_phone     = next_phone,
            prev_mfcc_exit = prev_mfcc_exit,
        )
        if meta is None:
            continue

        audio, _ = read_wav_mono(meta["path"])
        grain_arrays.append(audio)

        join_note = ""
        if args.strategy == "cost" and prev_mfcc_exit is not None and "mfcc_entry" in meta:
            dist = np.linalg.norm(prev_mfcc_exit - meta["mfcc_entry"])
            join_note = f"  join_dist={dist:.1f}"
        print(f"  {phone:6s}  {meta['path']}  ({meta['duration']:.3f}s){join_note}")

        prev_mfcc_exit = meta.get("mfcc_exit")  # carry forward for next join

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
