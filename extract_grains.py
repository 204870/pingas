#!/usr/bin/env python3
"""
extract_grains.py

Extracts phone-level audio grains from aligned speech for concatenative synthesis.

Supports two input formats:
  1. MFA TextGrid: --input-type textgrid --textgrid PATH --wav PATH
  2. BURNC corpus: --input-type burnc --speaker SPEAKER --burnc PATH

Grain boundaries mirror UTAU OTO concepts:
  offset       – start of the raw phone segment in the source (seconds)
  cutoff       – end of the raw phone segment (seconds)
  preutterance – how far before the vowel-onset the grain begins (seconds)
  overlap      – crossfade half-width shared with the next grain (seconds)

Each grain also carries phonetic context (prev_phone, next_phone, word) so
synth.py can do diphone/triphone-style context-aware selection.

Grains are written to  <output>/<PHONEME>/<index>.wav  (mono, 44100 Hz, int16).
A JSON index  <output>/index.json  lists every grain with its metadata.
"""

import argparse
import json
import re
import warnings
import wave
import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scipy.fft import dct

import aligned_textgrid as atg


# ---------------------------------------------------------------------------
# Warning filter for small boundary mismatches (< 10ms)
# ---------------------------------------------------------------------------

def small_mismatch_filter(message, category, filename, lineno, file=None, line=None):
    """Filter out aligned_textgrid warnings for mismatches < 10ms."""
    msg_str = str(message)
    match = re.search(r'largest mismatch was ([\d.]+)s', msg_str)
    if match:
        mismatch = float(match.group(1))
        if mismatch < 0.010:  # 10ms threshold
            return None  # Suppress
    return True  # Show

warnings.filterwarnings('default', category=UserWarning, module='aligned_textgrid')
old_showwarning = warnings.showwarning

def filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    if 'aligned_textgrid' in filename and category == UserWarning:
        result = small_mismatch_filter(message, category, filename, lineno, file, line)
        if result is None:
            return
    old_showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = filtered_showwarning
from praatio import textgrid as praatio_tg
from praatio.utilities.constants import Interval

# Import BURNC parser utilities
from pin.burnc_parser import (
    parse_label_file, parse_timit_phone, SKIP_PHONES
)

# ---------------------------------------------------------------------------
# OTO-style timing parameters (seconds)
# ---------------------------------------------------------------------------
PREUTTERANCE = 0.02   # look-back before phone onset  (consonant lead-in)
OVERLAP      = 0.015  # crossfade half-width at each boundary
MIN_DURATION = 0.02   # skip phones shorter than this (alignment noise)
FRAME_DUR    = 0.025  # MFCC analysis window width (seconds)

# ---------------------------------------------------------------------------
# TIMIT to ARPAbet phone mapping (for BURNC compatibility with synth.py)
# ---------------------------------------------------------------------------
TIMIT_TO_ARPABET = {
    # Vowels (mostly same, stress handled separately)
    'IY': 'IY', 'IH': 'IH', 'EH': 'EH', 'AE': 'AE', 'AH': 'AH', 'AO': 'AO',
    'UH': 'UH', 'UW': 'UW', 'AA': 'AA', 'ER': 'ER',
    'AX': 'AH', 'IX': 'IH', 'AXR': 'ER', 'UX': 'UW', 'AX-H': 'AH',
    # Diphthongs
    'EY': 'EY', 'AY': 'AY', 'OY': 'OY', 'AW': 'AW', 'OW': 'OW',
    # Stops
    'P': 'P', 'B': 'B', 'T': 'T', 'D': 'D', 'K': 'K', 'G': 'G',
    # Closures - skip (they're silence before stops)
    'PCL': None, 'BCL': None, 'TCL': None, 'DCL': None, 'KCL': None, 'GCL': None,
    # Affricates
    'CH': 'CH', 'JH': 'JH',
    # Fricatives
    'F': 'F', 'V': 'V', 'TH': 'TH', 'DH': 'DH', 'S': 'S', 'Z': 'Z',
    'SH': 'SH', 'ZH': 'ZH', 'HH': 'HH', 'HV': 'HH',
    # Nasals
    'M': 'M', 'N': 'N', 'NG': 'NG', 'EM': 'M', 'EN': 'N', 'ENG': 'NG', 'NX': 'N',
    # Liquids
    'L': 'L', 'R': 'R', 'EL': 'L', 'DX': 'D',
    # Glides
    'W': 'W', 'Y': 'Y', 'WH': 'W',
    # Silence/noise - skip
    'H#': None, 'PAU': None, 'brth': None, 'BRTH': None, 'epi': None, 'Q': None,
}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Return (samples float64 [-1,1], sample_rate). Stereo is averaged."""
    with wave.open(path) as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    peak = float(2 ** (8 * sampwidth - 1))
    samples /= peak

    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)

    return samples, sr


def write_wav_mono(path: str, samples: np.ndarray, sr: int) -> None:
    """Write float64 [-1,1] array as 16-bit mono PCM wav."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def sph_to_wav(sph_path: Path, wav_path: Path) -> bool:
    """Convert NIST SPHERE to WAV using sox."""
    try:
        subprocess.run(
            ['sox', str(sph_path), str(wav_path)],
            check=True, capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  sox error: {e.stderr.decode()[:100]}")
        return False


# ---------------------------------------------------------------------------
# MFCC helpers
# ---------------------------------------------------------------------------

def _mel_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """Return (n_filters, n_fft//2+1) triangular mel filterbank matrix."""
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    lo = hz_to_mel(80.0)
    hi = hz_to_mel(sr / 2.0)
    bins = np.floor(
        (n_fft + 1) * mel_to_hz(np.linspace(lo, hi, n_filters + 2)) / sr
    ).astype(int)

    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_filters, n_bins))
    for m in range(1, n_filters + 1):
        lo_b, mid_b, hi_b = bins[m - 1], bins[m], bins[m + 1]
        rise = max(mid_b - lo_b, 1)
        fall = max(hi_b - mid_b, 1)
        fb[m - 1, lo_b:mid_b] = (np.arange(lo_b, mid_b) - lo_b) / rise
        fb[m - 1, mid_b:hi_b] = (hi_b - np.arange(mid_b, hi_b)) / fall
    return fb


def compute_mfcc(samples: np.ndarray, sr: int, n_mfcc: int = 13, n_mel: int = 26) -> np.ndarray:
    """Return n_mfcc-dimensional MFCC vector for a short audio frame."""
    if len(samples) < 8:
        return np.zeros(n_mfcc)
    n_fft = max(512, 1 << int(np.ceil(np.log2(len(samples)))))
    emph = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    windowed = emph * np.hanning(len(emph))
    power = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    log_mel = np.log(_mel_filterbank(n_mel, n_fft, sr) @ power + 1e-8)
    return dct(log_mel, type=2, norm="ortho")[:n_mfcc]


def hann_fade(samples: np.ndarray, fade_len: int) -> np.ndarray:
    """Apply a Hann fade-in and fade-out of `fade_len` samples."""
    out = samples.copy()
    if fade_len < 1 or len(out) < 2 * fade_len:
        return out
    window = np.hanning(fade_len * 2)
    out[:fade_len] *= window[:fade_len]
    out[-fade_len:] *= window[fade_len:]
    return out


# ---------------------------------------------------------------------------
# Acoustic boundary refinement
# ---------------------------------------------------------------------------

# Phone classes for acoustic refinement and duration filtering
FRICATIVES = {'F', 'V', 'S', 'Z', 'SH', 'ZH', 'TH', 'DH', 'HH'}
STOPS = {'P', 'B', 'T', 'D', 'K', 'G'}
AFFRICATES = {'CH', 'JH'}
NASALS = {'M', 'N', 'NG'}
VOWELS = {'IY', 'IH', 'EH', 'AE', 'AH', 'AO', 'UH', 'UW', 'AA', 'ER'}
DIPHTHONGS = {'EY', 'AY', 'OY', 'AW', 'OW'}
APPROXIMANTS = {'L', 'R', 'W', 'Y'}

# Max duration limits (seconds) - reject alignment outliers
MAX_DURATION = {
    'vowel': 0.250,
    'diphthong': 0.300,
    'stop': 0.100,
    'affricate': 0.150,
    'fricative': 0.200,
    'nasal': 0.150,
    'approximant': 0.150,
}
MAX_DURATION_DEFAULT = 0.300


def get_phone_class(phone: str) -> str:
    """Return phone class for duration filtering."""
    p = phone.rstrip('012').upper()
    if p in VOWELS:
        return 'vowel'
    if p in DIPHTHONGS:
        return 'diphthong'
    if p in STOPS:
        return 'stop'
    if p in AFFRICATES:
        return 'affricate'
    if p in FRICATIVES:
        return 'fricative'
    if p in NASALS:
        return 'nasal'
    if p in APPROXIMANTS:
        return 'approximant'
    return 'default'


def get_max_duration(phone: str) -> float:
    """Return max allowed duration for a phone."""
    pclass = get_phone_class(phone)
    return MAX_DURATION.get(pclass, MAX_DURATION_DEFAULT)


# Refinement parameters
REFINE_WINDOW = 0.130  # ±130ms search window
REFINE_HOP = 0.010     # 10ms hop for feature computation


def compute_rms_envelope(samples: np.ndarray, sr: int, hop_sec: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMS energy envelope. Returns (times, rms)."""
    hop = int(hop_sec * sr)
    win = hop * 2
    n_frames = (len(samples) - win) // hop + 1
    if n_frames < 1:
        return np.array([0.0]), np.array([0.0])

    times = np.arange(n_frames) * hop_sec + (win / sr / 2)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        frame = samples[i * hop : i * hop + win]
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)
    return times, rms


def compute_spectral_tilt(samples: np.ndarray, sr: int, hop_sec: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral tilt (HF/LF energy ratio) for fricative detection.
    Higher values indicate more high-frequency energy (fricatives).
    """
    hop = int(hop_sec * sr)
    win = int(0.010 * sr)  # 10ms window
    n_fft = 512
    n_frames = (len(samples) - win) // hop + 1
    if n_frames < 1:
        return np.array([0.0]), np.array([0.0])

    times = np.arange(n_frames) * hop_sec + (win / sr / 2)
    tilt = np.zeros(n_frames)

    # Split at 3kHz for HF/LF
    split_bin = int(3000 * n_fft / sr)

    for i in range(n_frames):
        frame = samples[i * hop : i * hop + win]
        if len(frame) < win:
            continue
        windowed = frame * np.hanning(len(frame))
        spec = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
        lf_energy = np.sum(spec[:split_bin]) + 1e-10
        hf_energy = np.sum(spec[split_bin:]) + 1e-10
        tilt[i] = hf_energy / lf_energy

    return times, tilt


def compute_transient(samples: np.ndarray, sr: int, hop_sec: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute transient detection function for stop bursts.
    Uses spectral flux (increase in energy across frequency bands).
    """
    hop = int(hop_sec * sr)
    win = int(0.010 * sr)
    n_fft = 512
    n_frames = (len(samples) - win) // hop + 1
    if n_frames < 2:
        return np.array([0.0]), np.array([0.0])

    times = np.arange(n_frames) * hop_sec + (win / sr / 2)
    flux = np.zeros(n_frames)
    prev_spec = None

    for i in range(n_frames):
        frame = samples[i * hop : i * hop + win]
        if len(frame) < win:
            continue
        windowed = frame * np.hanning(len(frame))
        spec = np.abs(np.fft.rfft(windowed, n=n_fft))

        if prev_spec is not None:
            # Half-wave rectified spectral flux (only increases)
            diff = spec - prev_spec
            flux[i] = np.sum(np.maximum(0, diff))
        prev_spec = spec

    return times, flux


def compute_voicing(samples: np.ndarray, sr: int, hop_sec: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute voicing strength using autocorrelation peak.
    Higher values indicate periodic excitation (vowels/voiced sounds).
    """
    hop = int(hop_sec * sr)
    win = int(0.025 * sr)  # 25ms for pitch detection
    n_frames = (len(samples) - win) // hop + 1
    if n_frames < 1:
        return np.array([0.0]), np.array([0.0])

    times = np.arange(n_frames) * hop_sec + (win / sr / 2)
    voicing = np.zeros(n_frames)

    # Search for F0 between 80-400 Hz
    min_lag = int(sr / 400)
    max_lag = int(sr / 80)

    for i in range(n_frames):
        frame = samples[i * hop : i * hop + win]
        if len(frame) < win:
            continue

        # Normalized autocorrelation
        frame = frame - np.mean(frame)
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
            # Find peak in F0 range
            search = autocorr[min_lag:max_lag]
            if len(search) > 0:
                voicing[i] = np.max(search)

    return times, voicing


def refine_boundary(
    samples: np.ndarray,
    sr: int,
    boundary_sec: float,
    phone_label: str,
    is_onset: bool = True
) -> float:
    """
    Refine a phone boundary using acoustic features.

    Args:
        samples: Full audio array
        sr: Sample rate
        boundary_sec: Initial boundary time in seconds
        phone_label: ARPAbet phone label (without stress)
        is_onset: True if this is phone onset, False if offset

    Returns:
        Refined boundary time in seconds
    """
    phone_base = phone_label.rstrip('012').upper()

    # Extract window around boundary
    win_samples = int(REFINE_WINDOW * sr)
    center_sample = int(boundary_sec * sr)
    start_sample = max(0, center_sample - win_samples)
    end_sample = min(len(samples), center_sample + win_samples)

    window = samples[start_sample:end_sample]
    if len(window) < int(0.010 * sr):
        return boundary_sec

    window_start_sec = start_sample / sr

    # Choose detection method based on phone class
    if phone_base in FRICATIVES:
        # Use spectral tilt - fricatives have high HF energy
        times, feature = compute_spectral_tilt(window, sr)
        if is_onset:
            # Find steepest increase in HF energy
            deriv = np.diff(feature)
            if len(deriv) > 0:
                best_idx = np.argmax(deriv)
                return window_start_sec + times[best_idx]
        else:
            # Find steepest decrease
            deriv = np.diff(feature)
            if len(deriv) > 0:
                best_idx = np.argmin(deriv)
                return window_start_sec + times[best_idx]

    elif phone_base in STOPS or phone_base in AFFRICATES:
        # Use transient detection - stops have burst
        times, feature = compute_transient(window, sr)
        if is_onset:
            # Find maximum transient (burst)
            if len(feature) > 0:
                best_idx = np.argmax(feature)
                # Only refine if there's a significant transient
                if feature[best_idx] > np.mean(feature) + 2 * np.std(feature):
                    return window_start_sec + times[best_idx]
        else:
            # Offset: use energy drop
            times, rms = compute_rms_envelope(window, sr)
            deriv = np.diff(rms)
            if len(deriv) > 0:
                best_idx = np.argmin(deriv)
                return window_start_sec + times[best_idx]

    elif phone_base in VOWELS or phone_base in APPROXIMANTS:
        # Use voicing strength - vowels have periodic excitation
        times, feature = compute_voicing(window, sr)
        if is_onset:
            # Find where voicing strength increases
            deriv = np.diff(feature)
            if len(deriv) > 0:
                best_idx = np.argmax(deriv)
                return window_start_sec + times[best_idx]
        else:
            # Find where voicing drops
            deriv = np.diff(feature)
            if len(deriv) > 0:
                best_idx = np.argmin(deriv)
                return window_start_sec + times[best_idx]

    elif phone_base in NASALS:
        # Nasals: use voicing (they're voiced) + low frequency energy
        times, feature = compute_voicing(window, sr)
        if is_onset:
            deriv = np.diff(feature)
            if len(deriv) > 0:
                best_idx = np.argmax(deriv)
                return window_start_sec + times[best_idx]
        else:
            deriv = np.diff(feature)
            if len(deriv) > 0:
                best_idx = np.argmin(deriv)
                return window_start_sec + times[best_idx]

    # Fallback: use energy envelope
    times, rms = compute_rms_envelope(window, sr)
    deriv = np.diff(rms)
    if len(deriv) > 0:
        if is_onset:
            best_idx = np.argmax(deriv)
        else:
            best_idx = np.argmin(deriv)
        return window_start_sec + times[best_idx]

    return boundary_sec


# ---------------------------------------------------------------------------
# BURNC to TextGrid conversion
# ---------------------------------------------------------------------------

def timit_to_arpabet(timit_label: str) -> tuple[str | None, bool]:
    """
    Convert TIMIT phone label to ARPAbet.

    Returns: (arpabet_phone, stressed)
        - arpabet_phone: None if phone should be skipped (silence/closure)
        - stressed: True if phone had +1 marker
    """
    # Check for stress marker
    stressed = '+1' in timit_label
    base = timit_label.replace('+1', '').replace('+0', '')

    arpabet = TIMIT_TO_ARPABET.get(base)
    return arpabet, stressed


def burnc_to_textgrid(lba_path: Path, wrd_path: Path | None, max_time: float) -> praatio_tg.Textgrid:
    """
    Convert BURNC .lba (phones) and .wrd (words) to praatio TextGrid.

    Args:
        lba_path: Path to .lba file (phone alignments)
        wrd_path: Path to .wrd file (word alignments), or None
        max_time: Maximum time (audio duration)

    Returns:
        praatio Textgrid with words and phones tiers
    """
    # Parse phone labels
    lba_labels = parse_label_file(lba_path)

    # Convert to intervals with ARPAbet labels
    phone_intervals = []
    prev_end = 0.0

    for end_time, label in lba_labels:
        if label in SKIP_PHONES:
            prev_end = end_time
            continue

        arpabet, stressed = timit_to_arpabet(label)
        if arpabet is None:
            prev_end = end_time
            continue

        # Handle non-monotonic timestamps: if this interval starts after its end,
        # truncate the previous interval to end at this interval's start
        if prev_end >= end_time:
            prev_end = end_time - 0.001  # small epsilon to ensure valid interval
            if phone_intervals and phone_intervals[-1].end > prev_end:
                # Truncate previous interval
                last = phone_intervals[-1]
                phone_intervals[-1] = Interval(last.start, prev_end, last.label)

        # Skip if still invalid
        if prev_end >= end_time:
            continue

        # Add stress marker to vowels (ARPAbet convention)
        if stressed and arpabet in {'IY', 'IH', 'EH', 'AE', 'AH', 'AO', 'UH', 'UW',
                                     'AA', 'ER', 'EY', 'AY', 'OY', 'AW', 'OW'}:
            arpabet = arpabet + '1'

        phone_intervals.append(Interval(prev_end, end_time, arpabet))
        prev_end = end_time

    # Parse word labels if available
    word_intervals = []
    if wrd_path and wrd_path.exists():
        wrd_labels = parse_label_file(wrd_path)
        prev_end = 0.0
        for end_time, label in wrd_labels:
            if label and label not in SKIP_PHONES:
                # Handle non-monotonic timestamps
                if prev_end >= end_time:
                    prev_end = end_time - 0.001
                    if word_intervals and word_intervals[-1].end > prev_end:
                        last = word_intervals[-1]
                        word_intervals[-1] = Interval(last.start, prev_end, last.label)
                if prev_end < end_time:
                    word_intervals.append(Interval(prev_end, end_time, label))
            prev_end = end_time

    # Create TextGrid
    tg = praatio_tg.Textgrid()

    if word_intervals:
        word_tier = praatio_tg.IntervalTier('words', word_intervals, minT=0, maxT=max_time)
        tg.addTier(word_tier)

    phone_tier = praatio_tg.IntervalTier('phones', phone_intervals, minT=0, maxT=max_time)
    tg.addTier(phone_tier)

    return tg


def find_burnc_utterances(speaker_dir: Path) -> list[tuple[Path, Path, Path | None]]:
    """
    Find all (sph, lba, wrd) tuples for a BURNC speaker.

    Returns list of (sph_path, lba_path, wrd_path_or_none)
    """
    utterances = []
    for sph_path in speaker_dir.rglob('*.sph'):
        lba_path = sph_path.with_suffix('.lba')
        wrd_path = sph_path.with_suffix('.wrd')
        if lba_path.exists():
            utterances.append((sph_path, lba_path, wrd_path if wrd_path.exists() else None))
    return sorted(utterances)


# ---------------------------------------------------------------------------
# Grain extraction
# ---------------------------------------------------------------------------

def extract_grains_from_textgrid(
    tg: atg.AlignedTextGrid,
    audio: np.ndarray,
    sr: int,
    out_dir: Path,
    index: list,
    counts: dict,
    source_file: str = None,
    refine_boundaries: bool = True
) -> int:
    """
    Extract grains from an AlignedTextGrid.

    Args:
        refine_boundaries: If True, apply acoustic boundary refinement.

    Returns number of grains extracted.
    """
    total_dur = len(audio) / sr
    fade_samples = max(1, int(OVERLAP * sr))

    group = tg[0]
    # Find phones tier (might be index 0 if no words tier, or index 1 with words)
    phones_tier = group[-1]  # phones are always the last/deepest tier

    n_extracted = 0
    n_refined = 0
    n_rejected_duration = 0

    # Build list for manual prev/next context lookup
    phone_list = [p for p in phones_tier if p.label.strip()]

    for idx, phone in enumerate(phone_list):
        label = phone.label.strip()
        if not label:
            continue

        duration = phone.end - phone.start
        if duration < MIN_DURATION:
            continue

        # Reject outliers exceeding max duration for phone class
        max_dur = get_max_duration(phone.label)
        if duration > max_dur:
            n_rejected_duration += 1
            continue

        # Context: use global tier position, not atg's within-word linking
        prev_label = phone_list[idx - 1].label.strip() if idx > 0 else None
        next_label = phone_list[idx + 1].label.strip() if idx < len(phone_list) - 1 else None

        # Word context (if available)
        word_label = None
        if hasattr(phone, 'within') and phone.within:
            word_label = getattr(phone.within, 'label', '').strip() or None

        # Strip stress for directory name
        phone_base = label.rstrip('012')

        # Get phone boundaries
        phone_start = phone.start
        phone_end = phone.end

        # Apply acoustic boundary refinement
        if refine_boundaries:
            refined_start = refine_boundary(audio, sr, phone_start, phone_base, is_onset=True)
            refined_end = refine_boundary(audio, sr, phone_end, phone_base, is_onset=False)

            # Sanity check: refined boundaries should be close to original
            # and maintain minimum duration
            if abs(refined_start - phone_start) < REFINE_WINDOW:
                if refined_end - refined_start >= MIN_DURATION:
                    phone_start = refined_start
                    n_refined += 1
            if abs(refined_end - phone_end) < REFINE_WINDOW:
                if refined_end - phone_start >= MIN_DURATION:
                    phone_end = refined_end

        # Re-check duration after refinement
        refined_duration = phone_end - phone_start
        if refined_duration > max_dur or refined_duration < MIN_DURATION:
            n_rejected_duration += 1
            continue

        # OTO-style boundaries
        offset_sec = max(0.0, phone_start - PREUTTERANCE)
        cutoff_sec = min(total_dur, phone_end + OVERLAP)

        offset_samp = int(offset_sec * sr)
        cutoff_samp = int(cutoff_sec * sr)

        raw_grain = audio[offset_samp:cutoff_samp]
        if len(raw_grain) < 10:
            continue

        # MFCC at entry/exit
        half_frame = int(FRAME_DUR * sr) // 2
        onset_samp = int(PREUTTERANCE * sr)
        phone_end_samp = int((phone.end - offset_sec) * sr)

        entry_lo = max(0, onset_samp - half_frame)
        entry_hi = min(len(raw_grain), onset_samp + half_frame)
        mfcc_entry = compute_mfcc(raw_grain[entry_lo:entry_hi], sr).tolist()

        exit_lo = max(0, phone_end_samp - half_frame)
        exit_hi = min(len(raw_grain), phone_end_samp + half_frame)
        mfcc_exit = compute_mfcc(raw_grain[exit_lo:exit_hi], sr).tolist()

        grain = hann_fade(raw_grain, fade_samples)

        # Write grain
        phoneme_dir = out_dir / phone_base
        idx = counts.get(phone_base, 0)
        grain_path = phoneme_dir / f"{idx:04d}.wav"
        write_wav_mono(str(grain_path), grain, sr)
        counts[phone_base] = idx + 1

        index.append({
            "phoneme": phone_base,
            "phone_with_stress": label,
            "prev_phone": prev_label.rstrip('012') if prev_label else None,
            "next_phone": next_label.rstrip('012') if next_label else None,
            "word": word_label,
            "path": str(grain_path),
            "offset": offset_sec,
            "cutoff": cutoff_sec,
            "preutterance": PREUTTERANCE,
            "overlap": OVERLAP,
            "duration": phone_end - phone_start,
            "source_file": source_file,
            "source_start": phone_start,
            "source_end": phone_end,
            "original_start": phone.start,
            "original_end": phone.end,
            "mfcc_entry": mfcc_entry,
            "mfcc_exit": mfcc_exit,
        })
        n_extracted += 1

    if refine_boundaries and n_refined > 0:
        print(f"  Refined {n_refined} boundaries acoustically")
    if n_rejected_duration > 0:
        print(f"  Rejected {n_rejected_duration} grains exceeding max duration")

    return n_extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract phone-level grains for concatenative synthesis'
    )
    parser.add_argument('--input-type', choices=['textgrid', 'burnc'], default=None,
                        help='Input format: MFA textgrid or BURNC corpus (auto-detected)')

    # TextGrid mode options
    parser.add_argument('--textgrid', type=Path,
                        help='Path to TextGrid file (textgrid mode)')
    parser.add_argument('--wav', type=Path,
                        help='Path to WAV file (textgrid mode)')

    # BURNC mode options
    parser.add_argument('--speaker', type=str,
                        help='BURNC speaker ID, e.g., m1b, f1a (burnc mode)')
    parser.add_argument('--burnc', type=Path, default=Path('../bu_radio/data'),
                        help='Path to BURNC data directory')

    # Common options
    parser.add_argument('--output', type=Path, default=Path('data/grains'),
                        help='Output directory for grains')
    parser.add_argument('--target-sr', type=int, default=44100,
                        help='Target sample rate (default: 44100)')
    parser.add_argument('--no-refine', action='store_true',
                        help='Disable acoustic boundary refinement')
    parser.add_argument('--refine', action='store_true', default=True,
                        help='Enable acoustic boundary refinement (default)')

    args = parser.parse_args()

    # Auto-detect input type
    if args.input_type is None:
        if args.speaker:
            args.input_type = 'burnc'
        elif args.textgrid or args.wav:
            args.input_type = 'textgrid'
        else:
            parser.error('Specify --textgrid/--wav or --speaker/--burnc')

    # Ensure data/ directory exists
    os.makedirs('data', exist_ok=True)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    counts = {}
    total_grains = 0

    if args.input_type == 'textgrid':
        # MFA TextGrid mode
        if not args.textgrid or not args.wav:
            parser.error('--textgrid and --wav are required for textgrid mode')

        print(f"Loading {args.wav}")
        audio, sr = read_wav_mono(str(args.wav))
        print(f"  {len(audio)} samples | {sr} Hz | {len(audio)/sr:.2f}s")

        # Resample if needed
        if sr != args.target_sr:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * args.target_sr / sr))
            sr = args.target_sr
            print(f"  Resampled to {sr} Hz")

        print(f"Parsing {args.textgrid}")
        tg = atg.AlignedTextGrid(
            textgrid_path=str(args.textgrid),
            entry_classes=[atg.Word, atg.Phone],
        )

        refine = not args.no_refine
        total_grains = extract_grains_from_textgrid(
            tg, audio, sr, out_dir, index, counts,
            source_file=str(args.wav),
            refine_boundaries=refine
        )

    elif args.input_type == 'burnc':
        # BURNC corpus mode
        if not args.speaker:
            parser.error('--speaker is required for burnc mode')

        speaker_dir = args.burnc / args.speaker
        if not speaker_dir.exists():
            print(f"Speaker directory not found: {speaker_dir}")
            return

        print(f"Extracting grains from BURNC speaker {args.speaker}")
        print(f"Source: {speaker_dir}")
        print(f"Output: {out_dir}")

        utterances = find_burnc_utterances(speaker_dir)
        print(f"Found {len(utterances)} utterances with alignments")

        for i, (sph_path, lba_path, wrd_path) in enumerate(utterances):
            # Convert sph to wav
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_wav = Path(tmp.name)

            if not sph_to_wav(sph_path, tmp_wav):
                continue

            try:
                audio, sr = read_wav_mono(str(tmp_wav))
            finally:
                tmp_wav.unlink()

            # Resample if needed
            if sr != args.target_sr:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * args.target_sr / sr))
                sr = args.target_sr

            total_dur = len(audio) / sr

            # Convert to TextGrid
            praatio_tg_obj = burnc_to_textgrid(lba_path, wrd_path, total_dur)

            # Load with aligned_textgrid
            entry_classes = [atg.Word, atg.Phone] if wrd_path else [atg.Phone]
            tg = atg.AlignedTextGrid(
                textgrid=praatio_tg_obj,
                entry_classes=entry_classes
            )

            refine = not args.no_refine
            n = extract_grains_from_textgrid(
                tg, audio, sr, out_dir, index, counts,
                source_file=str(sph_path),
                refine_boundaries=refine
            )
            total_grains += n

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(utterances)} utterances, {total_grains} grains")

    # Write index
    index_path = out_dir / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\nExtracted {len(index)} grains across {len(counts)} phoneme classes.")
    print(f"Index written to {index_path}")
    for ph, n in sorted(counts.items()):
        print(f"  {ph:6s}  {n:4d} grains")


if __name__ == "__main__":
    main()
