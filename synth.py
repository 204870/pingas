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
from scipy.signal import butter, sosfilt, lfilter
from scipy.linalg import solve_toeplitz
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
# LPC Resynthesis with Ratio-Based Formant Targeting
# ---------------------------------------------------------------------------

# Vowel formant ratio templates (F2/F1, F3/F2)
# Based on Peterson & Barney (1952) averaged across speakers
VOWEL_RATIOS = {
    # Front vowels (high F2/F1)
    'IY': {'f2_f1': 8.0, 'f3_f2': 1.28},
    'IH': {'f2_f1': 5.4, 'f3_f2': 1.22},
    'EY': {'f2_f1': 5.8, 'f3_f2': 1.20},
    'EH': {'f2_f1': 3.2, 'f3_f2': 1.32},
    'AE': {'f2_f1': 2.6, 'f3_f2': 1.40},
    # Back vowels (low F2/F1)
    'AA': {'f2_f1': 1.5, 'f3_f2': 2.24},
    'AO': {'f2_f1': 1.4, 'f3_f2': 2.70},
    'OW': {'f2_f1': 1.8, 'f3_f2': 2.90},
    'UH': {'f2_f1': 2.2, 'f3_f2': 2.20},
    'UW': {'f2_f1': 2.5, 'f3_f2': 2.80},
    # Central/r-colored
    'AH': {'f2_f1': 1.9, 'f3_f2': 1.95},
    'ER': {'f2_f1': 2.0, 'f3_f2': 1.15},
    # Diphthongs (use starting position)
    'AY': {'f2_f1': 1.5, 'f3_f2': 2.20},
    'AW': {'f2_f1': 1.5, 'f3_f2': 2.20},
    'OY': {'f2_f1': 1.4, 'f3_f2': 2.70},
}

# How strongly to pull toward target ratios (0=none, 1=full snap)
FORMANT_TARGETING_STRENGTH = 0.1

# LPC parameters
LPC_ORDER = 16
LPC_WINDOW_SEC = 0.025
LPC_HOP_SEC = 0.010


def extract_formants_praat(
    samples: np.ndarray,
    sr: int,
    time: float = None,
    max_formant: float = 5500.0
) -> tuple[float, float, float] | None:
    """
    Extract F1, F2, F3 at a given time using Parselmouth.

    Args:
        samples: Audio samples
        sr: Sample rate
        time: Time point to extract (default: middle of signal)
        max_formant: Maximum formant frequency (5500 for male, 5500 for female)

    Returns:
        (F1, F2, F3) in Hz, or None if extraction fails
    """
    if not PARSELMOUTH_AVAILABLE:
        return None

    try:
        sound = parselmouth.Sound(samples, sampling_frequency=sr)
        if time is None:
            time = sound.duration / 2

        formant = call(sound, "To Formant (burg)", 0.01, 5, max_formant, 0.025, 50.0)

        f1 = call(formant, "Get value at time", 1, time, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, time, "Hertz", "Linear")
        f3 = call(formant, "Get value at time", 3, time, "Hertz", "Linear")

        if np.isnan(f1) or np.isnan(f2) or np.isnan(f3):
            return None

        return (f1, f2, f3)
    except:
        return None


def extract_lpc_praat(
    samples: np.ndarray,
    sr: int,
    order: int = LPC_ORDER
) -> 'parselmouth.LPC | None':
    """Extract LPC object using Parselmouth."""
    if not PARSELMOUTH_AVAILABLE:
        return None

    try:
        sound = parselmouth.Sound(samples, sampling_frequency=sr)
        lpc = call(sound, "To LPC (autocorrelation)", order, LPC_WINDOW_SEC, LPC_HOP_SEC, 50.0)
        return lpc
    except:
        return None


def lpc_to_source(sound: 'parselmouth.Sound', lpc: 'parselmouth.LPC') -> 'parselmouth.Sound':
    """Extract source/excitation by inverse filtering."""
    return call([sound, lpc], "Filter (inverse)")


def source_to_sound(source: 'parselmouth.Sound', lpc: 'parselmouth.LPC') -> 'parselmouth.Sound':
    """Resynthesize by filtering source through LPC."""
    return call([source, lpc], "Filter")


def shift_formants_to_ratio(
    samples: np.ndarray,
    sr: int,
    target_f2_f1: float,
    target_f3_f2: float,
    strength: float = FORMANT_TARGETING_STRENGTH
) -> np.ndarray:
    """
    Shift formants toward target ratios using LPC manipulation.

    Anchors on F1, adjusts F2 and F3 to approach target ratios.
    """
    if not PARSELMOUTH_AVAILABLE:
        return samples

    # Get current formants
    formants = extract_formants_praat(samples, sr)
    if formants is None:
        return samples

    f1, f2, f3 = formants

    if f1 < 50 or f2 < 50 or f3 < 50:
        return samples

    # Current ratios
    curr_f2_f1 = f2 / f1
    curr_f3_f2 = f3 / f2

    # Target ratios (interpolated by strength)
    new_f2_f1 = curr_f2_f1 + strength * (target_f2_f1 - curr_f2_f1)
    new_f3_f2 = curr_f3_f2 + strength * (target_f3_f2 - curr_f3_f2)

    # Compute shift factors
    # Keep F1 anchored, shift F2 and F3
    target_f2 = f1 * new_f2_f1
    target_f3 = target_f2 * new_f3_f2

    f2_shift = target_f2 / f2
    f3_shift = target_f3 / f3

    # Use Praat's formant shifting if shifts are significant
    if abs(f2_shift - 1.0) < 0.05 and abs(f3_shift - 1.0) < 0.05:
        return samples

    try:
        sound = parselmouth.Sound(samples, sampling_frequency=sr)

        # Use Change Gender which can shift formants
        # formant_shift_ratio shifts all formants proportionally
        # We approximate by using the average shift
        avg_shift = (f2_shift + f3_shift) / 2

        # Clamp to reasonable range
        avg_shift = max(0.7, min(1.4, avg_shift))

        shifted = call(sound, "Change gender", 75, 600, avg_shift, 0, 1.0, 1.0)
        return shifted.values[0]
    except:
        return samples


def lpc_resynth_grain(
    samples: np.ndarray,
    sr: int,
    phoneme: str | None = None,
    smooth_boundary_ms: float = 15.0
) -> tuple[np.ndarray, np.ndarray | None, 'parselmouth.LPC | None']:
    """
    Decompose grain into source + filter, optionally target vowel formants.

    Args:
        samples: Grain audio
        sr: Sample rate
        phoneme: ARPAbet phoneme (for formant targeting), or None
        smooth_boundary_ms: Fade length for boundary smoothing

    Returns:
        (processed_samples, source_signal, lpc_object)
    """
    if not PARSELMOUTH_AVAILABLE:
        return samples, None, None

    try:
        sound = parselmouth.Sound(samples, sampling_frequency=sr)

        # Extract LPC
        lpc = call(sound, "To LPC (autocorrelation)", LPC_ORDER, LPC_WINDOW_SEC, LPC_HOP_SEC, 50.0)

        # Extract source (excitation)
        source = call([sound, lpc], "Filter (inverse)")
        source_samples = source.values[0]

        # Apply formant targeting for vowels
        if phoneme:
            base_phoneme = re.sub(r'[0-9]', '', phoneme).upper()
            if base_phoneme in VOWEL_RATIOS:
                ratios = VOWEL_RATIOS[base_phoneme]
                samples = shift_formants_to_ratio(
                    samples, sr,
                    ratios['f2_f1'],
                    ratios['f3_f2'],
                    FORMANT_TARGETING_STRENGTH
                )

        return samples, source_samples, lpc
    except Exception as e:
        return samples, None, None


def lpc_analyze_frame(frame: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients for a single frame using autocorrelation method."""
    # Apply window
    windowed = frame * np.hanning(len(frame))

    # Autocorrelation
    n = len(windowed)
    r = np.correlate(windowed, windowed, mode='full')[n-1:n+order+1]

    # Levinson-Durbin recursion via scipy's solve_toeplitz
    if r[0] < 1e-10:
        return np.zeros(order)

    # Solve Toeplitz system: R @ a = r[1:]
    try:
        a = solve_toeplitz(r[:order], r[1:order+1])
        # Check stability: all poles must be inside unit circle
        # Quick check: if coefficients are too large, filter is unstable
        if np.any(np.abs(a) > 2.0) or np.any(np.isnan(a)):
            return np.zeros(order)
        return a
    except:
        return np.zeros(order)


def lpc_smooth_signal(
    samples: np.ndarray,
    sr: int,
    order: int = LPC_ORDER,
    window_sec: float = LPC_WINDOW_SEC,
    hop_sec: float = LPC_HOP_SEC,
    smooth_frames: int = 3
) -> np.ndarray:
    """
    Smooth signal by LPC analysis, coefficient smoothing, and resynthesis.

    This reduces formant discontinuities at grain boundaries by:
    1. Analyzing LPC coefficients frame-by-frame
    2. Applying temporal smoothing to coefficient trajectories
    3. Resynthesizing with smoothed filter
    """
    window_samp = int(window_sec * sr)
    hop_samp = int(hop_sec * sr)
    n_frames = (len(samples) - window_samp) // hop_samp + 1

    if n_frames < 2:
        return samples

    # Analyze: extract LPC coefficients for each frame
    lpc_coeffs = np.zeros((n_frames, order))
    for i in range(n_frames):
        start = i * hop_samp
        end = start + window_samp
        if end <= len(samples):
            lpc_coeffs[i] = lpc_analyze_frame(samples[start:end], order)

    # Smooth coefficient trajectories (simple moving average)
    smoothed_coeffs = np.copy(lpc_coeffs)
    for j in range(order):
        kernel = np.ones(smooth_frames) / smooth_frames
        smoothed_coeffs[:, j] = np.convolve(lpc_coeffs[:, j], kernel, mode='same')

    # Resynthesize: inverse filter with original, filter with smoothed
    output = np.zeros_like(samples)
    weight = np.zeros(len(samples))

    for i in range(n_frames):
        start = i * hop_samp
        end = start + window_samp
        if end > len(samples):
            break

        frame = samples[start:end]
        a_orig = lpc_coeffs[i]
        a_smooth = smoothed_coeffs[i]

        # Skip frames with zero/invalid coefficients
        if np.allclose(a_orig, 0) or np.allclose(a_smooth, 0):
            # Just copy original frame
            window = np.hanning(window_samp)
            output[start:end] += frame * window
            weight[start:end] += window
            continue

        # Inverse filter (get residual) using lfilter: residual = frame * A(z)
        # A(z) = 1 + a1*z^-1 + a2*z^-2 + ...
        a_fir = np.concatenate([[1.0], a_orig])
        residual = lfilter(a_fir, [1.0], frame)

        # Synthesis filter: synth = residual / A_smooth(z)
        # Use lfilter with IIR: b=1, a=[1, a_smooth]
        a_iir = np.concatenate([[1.0], a_smooth])
        synth = lfilter([1.0], a_iir, residual)

        # Check for numerical issues
        if np.any(np.isnan(synth)) or np.any(np.isinf(synth)) or np.max(np.abs(synth)) > 10:
            synth = frame  # Fall back to original

        # Overlap-add with Hann window
        window = np.hanning(window_samp)
        output[start:end] += synth * window
        weight[start:end] += window

    # Normalize by overlap weight
    weight = np.maximum(weight, 1e-8)
    output /= weight

    # Blend with original to preserve transients (70% original, 30% smoothed)
    blend = 0.7
    return blend * samples + (1 - blend) * output


def _fallback_ola_concat(grains: list[np.ndarray], overlap_samples: int) -> np.ndarray:
    """Simple OLA for fallback when Parselmouth unavailable."""
    if not grains:
        return np.array([], dtype=np.float64)
    if len(grains) == 1:
        return grains[0]
    total = sum(len(g) for g in grains) - overlap_samples * (len(grains) - 1)
    out = np.zeros(max(total, 1), dtype=np.float64)
    pos = 0
    for grain in grains:
        end = pos + len(grain)
        if end > len(out):
            out = np.concatenate([out, np.zeros(end - len(out))])
        out[pos:end] += grain
        pos += len(grain) - overlap_samples
    return out[:pos + overlap_samples]


def lpc_smooth_concat(
    grains: list[np.ndarray],
    phonemes: list[str],
    sr: int,
    overlap_samples: int,
    boundary_smooth_ms: float = 20.0
) -> np.ndarray:
    """
    Concatenate grains with LPC-based formant smoothing at boundaries.

    1. Decompose each grain into source + filter
    2. Apply formant targeting per phoneme
    3. Interpolate LPC at grain boundaries
    4. Resynthesize with smooth filter transitions
    """
    if not PARSELMOUTH_AVAILABLE or len(grains) == 0:
        # Fallback to simple OLA (defined later in this file)
        return _fallback_ola_concat(grains, overlap_samples)

    if len(grains) == 1:
        result, _, _ = lpc_resynth_grain(grains[0], sr, phonemes[0] if phonemes else None)
        return result

    # Process each grain
    processed_grains = []
    for i, (grain, phone) in enumerate(zip(grains, phonemes)):
        processed, _, _ = lpc_resynth_grain(grain, sr, phone)
        processed_grains.append(processed)

    # OLA concatenation with processed grains
    total = sum(len(g) for g in processed_grains) - overlap_samples * (len(processed_grains) - 1)
    total = max(total, sum(len(g) for g in processed_grains))
    out = np.zeros(total, dtype=np.float64)

    pos = 0
    for grain in processed_grains:
        end = pos + len(grain)
        if end > len(out):
            out = np.concatenate([out, np.zeros(end - len(out))])
        out[pos:end] += grain
        pos += len(grain) - overlap_samples

    return out[:pos + overlap_samples]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAIN_INDEX = "data/grains/index.json"
TG_PATH     = "aligned/bonfire.TextGrid"
OUT_WAV     = "data/out_synth.wav"
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

        # Apply boundary to last vowel (not last phone - F0 only meaningful on vowels)
        if word_info.boundary:
            # Find last vowel in this word's range, or in entire utterance for final boundary
            search_start = word_start if word_info != word_markers[-1] else 0
            last_vowel_idx = None
            for i in range(word_end - 1, search_start - 1, -1):
                if _is_vowel(phones[i]):
                    last_vowel_idx = i
                    break
            if last_vowel_idx is not None:
                boundary[last_vowel_idx] = True
                boundary_rising[last_vowel_idx] = word_info.boundary_rising

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

    # Spread boundary effects over the last half of the utterance
    f0_deltas = spread_boundary_contour(f0_deltas, boundary, boundary_rising)

    return f0_deltas


def spread_boundary_contour(
    f0_deltas: list[float],
    boundary: list[bool],
    boundary_rising: list[bool],
    phones: list[str] = None,
    spread_ratio: float = 0.5
) -> list[float]:
    """
    Spread boundary tone effects over the latter portion of the utterance.

    Instead of applying the boundary F0 change only to the final phone,
    create a gradual linear ramp over the last `spread_ratio` of the utterance,
    peaking at the last vowel.

    For falling boundaries (.), pitch gradually descends to the last vowel.
    For rising boundaries (?), pitch gradually rises to the last vowel.
    """
    n = len(f0_deltas)
    if n == 0:
        return f0_deltas

    # Find if there's a final boundary
    has_boundary = any(boundary)
    if not has_boundary:
        return f0_deltas

    # Find the last vowel (target for boundary tone - F0 is only meaningful on vowels)
    last_vowel_idx = None
    if phones:
        for i in range(n - 1, -1, -1):
            if _is_vowel(phones[i]):
                last_vowel_idx = i
                break

    if last_vowel_idx is None:
        return f0_deltas

    # Get the boundary delta from the model's prediction on the last vowel
    boundary_delta = f0_deltas[last_vowel_idx]

    # Calculate spread region: last half of utterance (or from start if short)
    spread_start = max(0, int(n * (1 - spread_ratio)))

    if last_vowel_idx <= spread_start:
        spread_start = max(0, last_vowel_idx - 1)

    result = f0_deltas.copy()

    # Linear ramp from spread_start to last_vowel_idx
    spread_length = last_vowel_idx - spread_start
    if spread_length <= 0:
        return result

    for i in range(spread_start, last_vowel_idx + 1):
        # Linear progress (0 at spread_start, 1 at last_vowel_idx)
        progress = (i - spread_start) / spread_length

        # Add the spread boundary effect (scaled linearly)
        result[i] += boundary_delta * progress * 0.5  # 50% of full effect spread

    return result


# ---------------------------------------------------------------------------
# F0 modification via Parselmouth
# ---------------------------------------------------------------------------

def apply_f0_modification(
    audio: np.ndarray,
    sr: int,
    phone_durations: list[float],
    f0_deltas: list[float],
) -> np.ndarray:
    """
    Apply F0 modifications to audio using Parselmouth PSOLA.

    Scales the existing pitch contour based on F0 deltas in cents.

    Args:
        audio: Input audio samples
        sr: Sample rate
        phone_durations: Duration of each phone in seconds
        f0_deltas: F0 delta in cents for each phone

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

    # Extract original pitch contour
    pitch = call(sound, "To Pitch", 0.0, 75, 600)

    # Create manipulation object
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    # Extract pitch tier from manipulation
    pitch_tier = call(manipulation, "Extract pitch tier")

    # Remove existing points
    n_points = call(pitch_tier, "Get number of points")
    for _ in range(int(n_points)):
        call(pitch_tier, "Remove point", 1)

    # Compute phone boundaries
    phone_times = []
    t = 0.0
    for dur in phone_durations:
        phone_times.append((t, t + dur))
        t += dur

    # Add modified pitch points for each phone
    for (t_start, t_end), delta_cents in zip(phone_times, f0_deltas):
        t_mid = (t_start + t_end) / 2

        try:
            # Get original F0 at this point
            original_f0 = call(pitch, "Get value at time", t_mid, "Hertz", "Linear")

            # Skip unvoiced regions
            if original_f0 == 0 or np.isnan(original_f0):
                continue

            # Apply delta: new_f0 = old_f0 * 2^(delta_cents / 1200)
            ratio = 2.0 ** (delta_cents / 1200.0)
            new_f0 = original_f0 * ratio

            # Clamp to reasonable range
            new_f0 = max(75, min(600, new_f0))

            call(pitch_tier, "Add point", t_mid, new_f0)
        except Exception:
            pass  # Skip if point can't be added

    # Replace pitch tier
    call([manipulation, pitch_tier], "Replace pitch tier")

    # Resynthesize using PSOLA
    result = call(manipulation, "Get resynthesis (overlap-add)")

    # Convert back to numpy array
    return result.values[0]

# ---------------------------------------------------------------------------
# H&B-style cost weights
# ---------------------------------------------------------------------------
W_TARGET      = 2.0   # weight for phonetic context mismatch (reduced for smoother joins)
W_JOIN        = 0.5   # weight for spectral join cost (prioritize smooth concatenation)
MFCC_SCALE    = 12.0  # normalising divisor for MFCC Euclidean distance
               #   lower = more sensitive to spectral mismatch at joins
STRESS_WEIGHT = 0.3   # penalty for vowel stress mismatch (autosegmental prominence)
               #   reduced to prioritize prosodic smoothness over stress matching

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
    # Include base vowel without stress marker (e.g., "EY" from BURNC)
    candidates.extend(mono.get(base, []))
    # Include all stress variants (e.g., "EY0", "EY1", "EY2")
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
    outlier_threshold: float = 2.5,
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
    # Check if target is a stressed vowel (prefer longer grains)
    target_stressed = _is_vowel(target_phoneme) and _vowel_stress(target_phoneme) >= 1

    # Compute duration statistics for outlier detection and stress bonus
    durations = np.array([g["duration"] for g in candidates])
    dur_mean = np.mean(durations)
    dur_std = np.std(durations)
    min_dur = np.min(durations)
    max_dur = np.max(durations)
    dur_range = max_dur - min_dur if max_dur > min_dur else 1.0

    def cost(g: dict) -> float:
        # target cost: context match (0–1)
        ctx = sum([g.get("prev_phone") == prev_phone,
                   g.get("next_phone") == next_phone])
        t_cost = (2 - ctx) / 2

        # stress prominence penalty (AM theory): 0 = exact, 0.5 = adjacent, 1.0 = max
        t_cost += STRESS_WEIGHT * _stress_cost(target_phoneme, g["phoneme"])

        # optional duration penalty, normalised to ~0–1
        if target_dur is not None:
            t_cost += 0.2 * abs(g["duration"] - target_dur) / max(target_dur, 1e-6)

        # Duration bonus for stressed vowels: prefer longer grains
        # Bonus ranges from 0 (shortest) to -0.5 (longest)
        if target_stressed:
            dur_normalized = (g["duration"] - min_dur) / dur_range
            t_cost -= 0.5 * dur_normalized  # negative cost = bonus

        # Outlier penalty: penalize grains with anomalous durations
        # (likely segmentation errors)
        if dur_std > 0:
            z_score = abs(g["duration"] - dur_mean) / dur_std
            if z_score > outlier_threshold:
                t_cost += z_score - outlier_threshold  # penalty grows with deviation

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
    strategy: str = "cost",  # H&B cost-minimising selection
    target_dur: float | None = None,
    prev_phone: str | None = None,
    next_phone: str | None = None,
    prev_mfcc_exit: np.ndarray | None = None,
) -> dict | None:
    """Pick one grain metadata dict for *phoneme* according to *strategy*."""
    candidates = _expanded_candidates(mono, phoneme)
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
        return _select_by_cost(candidates, phoneme, prev_phone, next_phone,
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
                        default="cost", help="Grain selection strategy (default: cost)")
    parser.add_argument("--overlap", type=float, default=0.015,
                        help="Crossfade half-width in seconds (default: 0.015)")
    parser.add_argument("--out", type=str, default=OUT_WAV,
                        help=f"Output wav path (default: {OUT_WAV})")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-prosody", action="store_true",
                        help="Disable prosody modification (F0 changes)")
    parser.add_argument("--lpc", action="store_true",
                        help="Enable LPC resynthesis with formant smoothing")
    parser.add_argument("--formant-strength", type=float, default=0.1,
                        help="Formant targeting strength 0-1 (default: 0.1)")
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
        # Strip stress for context matching (grains store context without stress)
        prev_phone = re.sub(r'[0-9]', '', phones[i - 1]) if i > 0 else None
        next_phone = re.sub(r'[0-9]', '', phones[i + 1]) if i < len(phones) - 1 else None

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
    if args.lpc and PARSELMOUTH_AVAILABLE:
        global FORMANT_TARGETING_STRENGTH
        FORMANT_TARGETING_STRENGTH = args.formant_strength
        print(f"\nUsing LPC resynthesis (formant strength={args.formant_strength})...")
        result = lpc_smooth_concat(grain_arrays, selected_phones, sr, overlap_samp)
    else:
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
        result = apply_f0_modification(result, sr, grain_durations, f0_deltas)
        print("  F0 modification applied")

    # -- high-pass filter (100 Hz, 8th-order Butterworth = -48 dB/oct) -------
    sos    = butter(8, 100.0, btype="high", fs=sr, output="sos")
    result = sosfilt(sos, result)

    # -- LPC boundary smoothing pass (numpy-based) ---------------------------
    if args.lpc:
        try:
            result = lpc_smooth_signal(result, sr)
            print("  LPC boundary smoothing applied")
        except Exception as e:
            print(f"  [warn] LPC smoothing skipped: {e}")

    # -- write output --------------------------------------------------------
    write_wav_mono(args.out, result, sr)
    print(f"\nWrote {len(result)/sr:.3f}s  →  {args.out}")


if __name__ == "__main__":
    main()
