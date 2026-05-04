#!/usr/bin/env python3
"""
synth.py

Granular concatenative speech synthesiser with prosodic control.

Given orthographic text with prosodic markers, this script:

  1. Parses prosodic markers (* . ? !) from input text.
  2. Resolves the input to a flat ARPAbet phone sequence (via g2p.py).
  3. Loads the grain index produced by extract_grains.py.
  4. For each phoneme, selects a grain from the corpus.
  5. Concatenates grains using overlap-add (OLA) crossfading.
  6. Applies F0 modifications based on prosody model predictions (Parselmouth PSOLA).
  7. (Future: LPC resynthesis for formant correction)
  8. Writes the result to an output wav.

Prosodic markers
----------------
  *word  — prominence (pitch accent): raises F0 on stressed vowel
  word.  — falling boundary (declarative): lowers F0
  word?  — rising boundary (question): raises F0
  word!  — emphatic (prominence + falling boundary)

Usage examples
--------------
  # Prosodic markers in text
  python synth.py "*hello world."
  python synth.py "is this *working?"
  python synth.py "*wow!"

  # H&B-style cost-minimising grain selection
  python synth.py "the quick brown fox" --strategy cost

  # bypass G2P with an explicit ARPAbet sequence
  python synth.py --phones "HH EH L OW"

  # disable prosody modification
  python synth.py "hello world" --no-prosody
"""

import argparse
import json
import os
import re
import random
import wave
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfilt
import aligned_textgrid as atg
from g2p import text_to_phones

# Prosody model imports (optional - graceful fallback if not available)
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("[warn] parselmouth not installed — prosody modification disabled")

try:
    from tensorflow.keras.models import load_model
    from pin.burnc_parser import phones_to_features, parse_arpabet_phone, cents_to_hz, hz_to_cents
    PROSODY_MODEL_AVAILABLE = True
except ImportError:
    PROSODY_MODEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAIN_INDEX = "grains/index.json"
TG_PATH     = "aligned/bonfire.TextGrid"
OUT_WAV     = "out_synth.wav"
PROSODY_MODEL_PATH = "pin/prosody_model.h5"
PROSODY_META_PATH  = "pin/prosody_model.json"


# ---------------------------------------------------------------------------
# Prosodic marker parsing
# ---------------------------------------------------------------------------

@dataclass
class ProsodyMarkers:
    """Prosodic annotations for a word."""
    word: str
    prominent: bool = False      # * before word
    boundary: bool = False       # . ? ! after word
    boundary_rising: bool = False  # ? = rising, . ! = falling


def parse_prosodic_text(text: str) -> list[ProsodyMarkers]:
    """
    Parse prosodic markers from input text.

    Markers:
      *word  — prominence (pitch accent)
      word.  — falling boundary
      word?  — rising boundary
      word!  — emphatic (prominence + falling)

    Returns list of ProsodyMarkers, one per word.
    """
    # Split on whitespace, preserving punctuation attached to words
    tokens = text.split()
    result = []

    for token in tokens:
        prominent = False
        boundary = False
        boundary_rising = False

        # Check for prominence marker at start
        if token.startswith('*'):
            prominent = True
            token = token[1:]

        # Check for boundary markers at end
        if token.endswith('.'):
            boundary = True
            boundary_rising = False
            token = token[:-1]
        elif token.endswith('?'):
            boundary = True
            boundary_rising = True
            token = token[:-1]
        elif token.endswith('!'):
            # Emphatic: prominence + falling boundary
            prominent = True
            boundary = True
            boundary_rising = False
            token = token[:-1]

        if token:  # Skip empty tokens
            result.append(ProsodyMarkers(
                word=token,
                prominent=prominent,
                boundary=boundary,
                boundary_rising=boundary_rising
            ))

    return result


def assign_prosody_to_phones(
    phones: list[str],
    word_markers: list[ProsodyMarkers]
) -> tuple[list[bool], list[bool], list[bool]]:
    """
    Assign prosodic features to phones based on word-level markers.

    For prominence: mark the stressed vowel of prominent words.
    For boundary: mark the last phone of boundary words.

    Returns: (prominent, boundary, boundary_rising) lists aligned with phones.
    """
    n = len(phones)
    prominent = [False] * n
    boundary = [False] * n
    boundary_rising = [False] * n

    if not word_markers:
        return prominent, boundary, boundary_rising

    # We need to align words to phones. This is approximate since we don't
    # have exact word boundaries. We'll distribute phones across words
    # proportionally based on typical phone counts.

    # Estimate phones per word (rough heuristic)
    total_words = len(word_markers)
    phones_per_word = n // max(total_words, 1)

    phone_idx = 0
    for word_info in word_markers:
        # Estimate phone range for this word
        word_start = phone_idx
        word_end = min(phone_idx + phones_per_word, n)

        # For last word, take remaining phones
        if word_info == word_markers[-1]:
            word_end = n

        if word_start >= n:
            break

        # Apply prominence to stressed vowel in this word's phone range
        if word_info.prominent:
            for i in range(word_start, word_end):
                phone = phones[i]
                # Check if it's a stressed vowel (ends in 1 or 2)
                if phone and phone[-1] in ('1', '2'):
                    prominent[i] = True
                    break  # Only mark first stressed vowel

        # Apply boundary to last phone of word
        if word_info.boundary:
            boundary[word_end - 1] = True
            boundary_rising[word_end - 1] = word_info.boundary_rising

        phone_idx = word_end

    return prominent, boundary, boundary_rising


# ---------------------------------------------------------------------------
# Prosody model inference
# ---------------------------------------------------------------------------

_prosody_model = None
_prosody_meta = None


def load_prosody_model():
    """Load the prosody model and metadata (cached)."""
    global _prosody_model, _prosody_meta

    if _prosody_model is not None:
        return _prosody_model, _prosody_meta

    if not PROSODY_MODEL_AVAILABLE:
        return None, None

    model_path = Path(PROSODY_MODEL_PATH)
    meta_path = Path(PROSODY_META_PATH)

    if not model_path.exists():
        print(f"[warn] Prosody model not found at {model_path}")
        return None, None

    _prosody_model = load_model(model_path, compile=False)

    if meta_path.exists():
        with open(meta_path) as f:
            _prosody_meta = json.load(f)
    else:
        _prosody_meta = {}

    return _prosody_model, _prosody_meta


def predict_f0_deltas(
    phones: list[str],
    prominent: list[bool],
    boundary: list[bool],
    boundary_rising: list[bool]
) -> list[float]:
    """
    Predict F0 deltas for each phone using the prosody model.

    Returns list of F0 deltas in cents.
    """
    model, meta = load_prosody_model()
    if model is None:
        return [0.0] * len(phones)

    # Build feature matrix
    X = phones_to_features(phones, prominent, boundary, boundary_rising)

    # Model expects (batch, seq_len, features)
    seq_len = meta.get('sequence_length', 200)
    X_padded = np.zeros((1, seq_len, X.shape[1]), dtype=np.float32)
    X_padded[0, :len(phones)] = X

    # Predict
    y_pred = model.predict(X_padded, verbose=0)

    # Extract F0 deltas (first output channel)
    f0_deltas = y_pred[0, :len(phones), 0].tolist()

    return f0_deltas


# ---------------------------------------------------------------------------
# F0 modification via Parselmouth
# ---------------------------------------------------------------------------

def apply_f0_modification(
    audio: np.ndarray,
    sr: int,
    phone_durations: list[float],
    f0_deltas: list[float],
    reference_f0: float = 150.0
) -> np.ndarray:
    """
    Apply F0 modifications to audio using Parselmouth PSOLA.

    Args:
        audio: Input audio samples
        sr: Sample rate
        phone_durations: Duration of each phone in seconds
        f0_deltas: F0 delta in cents for each phone
        reference_f0: Reference F0 for cents conversion (speaker mean)

    Returns:
        Modified audio samples
    """
    if not PARSELMOUTH_AVAILABLE:
        return audio

    if len(f0_deltas) != len(phone_durations):
        print("[warn] f0_deltas and phone_durations length mismatch")
        return audio

    # Create Parselmouth Sound object
    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    # Create manipulation object
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    # Get pitch tier
    pitch_tier = call(manipulation, "Extract pitch tier")

    # Compute phone boundaries
    phone_times = []
    t = 0.0
    for dur in phone_durations:
        phone_times.append((t, t + dur))
        t += dur

    # Add pitch points for each phone
    for (t_start, t_end), delta_cents in zip(phone_times, f0_deltas):
        if abs(delta_cents) < 1.0:  # Skip negligible changes
            continue

        t_mid = (t_start + t_end) / 2

        # Get current F0 at this point (if voiced)
        try:
            # Convert delta to ratio
            # delta_cents = 1200 * log2(new_f0 / old_f0)
            # new_f0 = old_f0 * 2^(delta_cents / 1200)
            ratio = 2.0 ** (delta_cents / 1200.0)

            # Add pitch point (Parselmouth uses Hz)
            # We'll modify relative to reference
            new_f0 = reference_f0 * ratio
            call(pitch_tier, "Add point", t_mid, new_f0)
        except Exception:
            pass  # Skip if point can't be added

    # Replace pitch tier
    call([manipulation, pitch_tier], "Replace pitch tier")

    # Resynthesize
    result = call(manipulation, "Get resynthesis (overlap-add)")

    # Convert back to numpy array
    return result.values[0]

# ---------------------------------------------------------------------------
# H&B-style cost weights
# ---------------------------------------------------------------------------
W_TARGET      = 1.0   # weight for phonetic context mismatch  (range 0–1)
W_JOIN        = 1.0   # weight for spectral join cost
MFCC_SCALE    = 20.0  # normalising divisor for MFCC Euclidean distance
               #   ~5–15 = within-class variation; ~20–40 = cross-class
STRESS_WEIGHT = 0.5   # penalty for vowel stress mismatch (autosegmental prominence)
               #   0 = exact, 0.5 = adjacent level (1↔2 or 0↔1), 1.0 = max (0↔2)

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
# Stress / prominence helpers  (autosegmental-metrical theory)
# ---------------------------------------------------------------------------

def _is_vowel(phoneme: str) -> bool:
    """True if phoneme carries an ARPAbet stress digit, i.e. is a vowel."""
    return bool(re.search(r"[0-9]", phoneme))


def _vowel_stress(phoneme: str) -> int:
    """Return ARPAbet stress digit (0/1/2). Call only after _is_vowel check."""
    return int(re.search(r"[0-9]", phoneme).group())


def _stress_cost(target_ph: str, candidate_ph: str) -> float:
    """
    Metrical prominence penalty: distance on the stress scale, normalised to [0, 1].
    0 = exact match or consonant, 0.5 = adjacent level, 1.0 = max mismatch (0↔2).
    """
    if not _is_vowel(target_ph) or not _is_vowel(candidate_ph):
        return 0.0
    return abs(_vowel_stress(target_ph) - _vowel_stress(candidate_ph)) / 2.0


def _expanded_candidates(mono: dict, phoneme: str) -> list[dict]:
    """
    For vowels, pool all stress-level variants of the same base phoneme.
    For consonants (no stress digit), return the exact entry only.
    """
    if not _is_vowel(phoneme):
        return mono.get(phoneme, [])
    base = re.sub(r"[0-9]", "", phoneme)
    candidates: list[dict] = []
    for stress in ("0", "1", "2"):
        candidates.extend(mono.get(base + stress, []))
    return candidates


# ---------------------------------------------------------------------------
# Grain selection strategies
# ---------------------------------------------------------------------------

def _select_by_cost(
    candidates: list[dict],
    target_phoneme: str,
    prev_phone: str | None,
    next_phone: str | None,
    prev_mfcc_exit: np.ndarray | None,
    target_dur: float | None,
) -> dict:
    """
    Simplified Hunt & Black (1996) cost: target_cost + join_cost.

    target_cost  — phonetic context mismatch (0 = perfect triphone,
                   0.5 = one neighbour wrong, 1.0 = neither matches)
                   plus stress prominence penalty and optional duration penalty.
    join_cost    — normalised MFCC Euclidean distance between the exit
                   frame of the previous grain and the entry frame of
                   this candidate (0 when no previous grain exists).
    """
    def cost(g: dict) -> float:
        # target cost: context match (0–1)
        ctx = sum([g.get("prev_phone") == prev_phone,
                   g.get("next_phone") == next_phone])
        t_cost = (2 - ctx) / 2

        # stress prominence penalty (AM theory): 0 = exact, 0.5 = adjacent, 1.0 = max
        t_cost += STRESS_WEIGHT * _stress_cost(target_phoneme, g["phoneme"])

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
    strategy: str = "cost",
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
    parser = argparse.ArgumentParser(description="Granular speech synthesiser with prosody")
    parser.add_argument("text", nargs="?", default=None,
                        help="Text with prosodic markers (*word. word? word!) to synthesise")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--phones", type=str,
                     help="ARPAbet phone sequence, e.g. 'HH EH L OW' (overrides text, no prosody)")
    src.add_argument("--word",   type=str,
                     help="Single word to look up in the TextGrid (overrides text, no prosody)")
    parser.add_argument("--durations", type=str, default=None,
                        help="Target grain durations in seconds, space-separated")
    parser.add_argument("--strategy", choices=["random", "longest", "shortest", "nearest", "cost"],
                        default="random", help="Grain selection strategy (default: random)")
    parser.add_argument("--overlap", type=float, default=0.015,
                        help="Crossfade half-width in seconds (default: 0.015)")
    parser.add_argument("--out", type=str, default=OUT_WAV,
                        help=f"Output wav path (default: {OUT_WAV})")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-prosody", action="store_true",
                        help="Disable prosody modification (F0 changes)")
    parser.add_argument("--reference-f0", type=float, default=150.0,
                        help="Reference F0 in Hz for prosody modification (default: 150)")
    args = parser.parse_args()

    if not any([args.text, args.phones, args.word]):
        parser.error("provide text to synthesise, --phones, or --word")

    if args.seed is not None:
        random.seed(args.seed)

    # -- load corpus ---------------------------------------------------------
    print(f"Loading grain index from {GRAIN_INDEX} …")
    mono, context_index = load_index(GRAIN_INDEX)
    print(f"  {sum(len(v) for v in mono.values())} grains  |  {len(mono)} phoneme classes")

    # -- parse prosodic markers and resolve phone sequence -------------------
    word_markers: list[ProsodyMarkers] = []
    use_prosody = not args.no_prosody and PARSELMOUTH_AVAILABLE

    if args.phones:
        phones = args.phones.split()
        print(f"  phones: {phones}")
        use_prosody = False  # No prosody for raw phone input
    elif args.word:
        phones = phones_for_word(args.word)
        print(f"  '{args.word}' → {phones}")
        use_prosody = False  # No prosody for word lookup
    else:
        # Parse prosodic markers from text
        word_markers = parse_prosodic_text(args.text)
        clean_text = " ".join(m.word for m in word_markers)
        print(f"  text: '{args.text}'")
        print(f"  clean: '{clean_text}'")

        # Show parsed prosody
        for m in word_markers:
            markers = []
            if m.prominent:
                markers.append("*")
            if m.boundary:
                markers.append("?" if m.boundary_rising else ".")
            if markers:
                print(f"    {m.word}: {' '.join(markers)}")

        phones = text_to_phones(clean_text)
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
    grain_durations: list[float] = []
    selected_phones: list[str] = []
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
        grain_durations.append(meta["duration"])
        selected_phones.append(phone)

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

    # -- prosody modification (F0) -------------------------------------------
    if use_prosody and word_markers and PROSODY_MODEL_AVAILABLE:
        print("\nApplying prosody modification...")

        # Assign prosodic features to phones
        prominent, boundary, boundary_rising = assign_prosody_to_phones(
            selected_phones, word_markers
        )

        # Predict F0 deltas
        f0_deltas = predict_f0_deltas(
            selected_phones, prominent, boundary, boundary_rising
        )

        # Show predictions for marked phones
        for i, (ph, prom, bnd, rising, delta) in enumerate(
            zip(selected_phones, prominent, boundary, boundary_rising, f0_deltas)
        ):
            if prom or bnd:
                marker = ""
                if prom:
                    marker += "*"
                if bnd:
                    marker += "?" if rising else "."
                print(f"    {ph:6s} [{marker}]: {delta:+.0f} cents")

        # Apply F0 modification via Parselmouth
        result = apply_f0_modification(
            result, sr, grain_durations, f0_deltas,
            reference_f0=args.reference_f0
        )
        print("  F0 modification applied")

    # -- high-pass filter (100 Hz, 8th-order Butterworth = -48 dB/oct) -------
    sos    = butter(8, 100.0, btype="high", fs=sr, output="sos")
    result = sosfilt(sos, result)

    # -- write output --------------------------------------------------------
    write_wav_mono(args.out, result, sr)
    print(f"\nWrote {len(result)/sr:.3f}s  →  {args.out}")


if __name__ == "__main__":
    main()
