#!/usr/bin/env python3
"""
extract_grains.py

Reads aligned/bonfire.TextGrid and bonfire/bonfire.wav, then extracts one
audio grain per phoneme interval.  Grain boundaries mirror UTAU OTO concepts:

  offset       – start of the raw phone segment in the source (seconds)
  cutoff       – end of the raw phone segment (seconds)
  preutterance – how far before the vowel-onset the grain begins (seconds)
                 here: fixed look-back into the preceding phone
  overlap      – crossfade half-width shared with the next grain (seconds)

Each grain also carries phonetic context (prev_phone, next_phone, word) so
synth.py can do diphone/triphone-style context-aware selection, matching the
classic concatenative approach but operating on arbitrary granular units.

Grains are written to  grains/<PHONEME>/<index>.wav  (mono, 44100 Hz, int16).
A JSON index  grains/index.json  lists every grain with its metadata.
"""

import json
import wave
import os
import numpy as np
from scipy.fft import dct
import aligned_textgrid as atg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TG_PATH  = "aligned/bonfire.TextGrid"
WAV_PATH = "bonfire/bonfire.wav"
OUT_DIR  = "grains"

# ---------------------------------------------------------------------------
# OTO-style timing parameters (seconds)
# ---------------------------------------------------------------------------
PREUTTERANCE = 0.02   # look-back before phone onset  (consonant lead-in)
OVERLAP      = 0.015  # crossfade half-width at each boundary
MIN_DURATION = 0.02   # skip phones shorter than this (alignment noise)
FRAME_DUR    = 0.025  # MFCC analysis window width (seconds)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Return (samples float64 [-1,1], sample_rate).  Stereo is averaged."""
    with wave.open(path) as wf:
        sr        = wf.getframerate()
        n_frames  = wf.getnframes()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw       = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    peak    = float(2 ** (8 * sampwidth - 1))
    samples /= peak

    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)

    return samples, sr


def write_wav_mono(path: str, samples: np.ndarray, sr: int) -> None:
    """Write float64 [-1,1] array as 16-bit mono PCM wav."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm     = (clipped * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _mel_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """Return (n_filters, n_fft//2+1) triangular mel filterbank matrix."""
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    lo   = hz_to_mel(80.0)
    hi   = hz_to_mel(sr / 2.0)
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
    n_fft    = max(512, 1 << int(np.ceil(np.log2(len(samples)))))
    emph     = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    windowed = emph * np.hanning(len(emph))
    power    = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    log_mel  = np.log(_mel_filterbank(n_mel, n_fft, sr) @ power + 1e-8)
    return dct(log_mel, type=2, norm="ortho")[:n_mfcc]


def hann_fade(samples: np.ndarray, fade_len: int) -> np.ndarray:
    """Apply a Hann fade-in and fade-out of `fade_len` samples."""
    out = samples.copy()
    if fade_len < 1 or len(out) < 2 * fade_len:
        return out
    window = np.hanning(fade_len * 2)
    out[:fade_len]  *= window[:fade_len]   # fade-in
    out[-fade_len:] *= window[fade_len:]   # fade-out
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- load audio ----------------------------------------------------------
    print(f"Loading {WAV_PATH} …")
    audio, sr = read_wav_mono(WAV_PATH)
    total_dur  = len(audio) / sr
    print(f"  {len(audio)} samples  |  {sr} Hz  |  {total_dur:.2f} s")

    # -- parse TextGrid with aligned_textgrid (gives word→phone hierarchy) --
    print(f"Parsing {TG_PATH} …")
    tg = atg.AlignedTextGrid(
        textgrid_path=TG_PATH,
        entry_classes=[atg.Word, atg.Phone],
    )
    group       = tg[0]
    phones_tier = group[1]  # "phones" IntervalTier (Phone SequenceIntervals)

    # -- extract grains ------------------------------------------------------
    index   = []   # list of grain metadata dicts
    counts  = {}   # phoneme → number of grains written so far

    fade_samples = max(1, int(OVERLAP * sr))

    for phone in phones_tier:
        label = phone.label.strip()
        if not label:
            continue  # skip silence/empty

        duration = phone.end - phone.start
        if duration < MIN_DURATION:
            continue

        # atg gives us .prev and .fol for neighbouring phones in the sequence;
        # boundary objects have empty labels, which we normalise to None.
        prev_label = phone.prev.label.strip() or None
        next_label = phone.fol.label.strip()  or None

        # parent word via the hierarchical containment link
        word_label = getattr(phone.within, "label", "").strip() or None

        # OTO-style boundaries (clamped to audio extent)
        offset_sec  = max(0.0,       phone.start - PREUTTERANCE)
        cutoff_sec  = min(total_dur, phone.end   + OVERLAP)

        offset_samp = int(offset_sec * sr)
        cutoff_samp = int(cutoff_sec * sr)

        raw_grain = audio[offset_samp:cutoff_samp]

        # MFCC snapshots at join boundaries, taken from raw audio before fading
        # so the spectral content isn't distorted by the envelope shaping.
        # entry = frame at the phone onset (where the preceding grain hands off)
        # exit  = frame at the phone end (where the following grain picks up)
        half_frame     = int(FRAME_DUR * sr) // 2
        onset_samp     = int(PREUTTERANCE * sr)               # within grain
        phone_end_samp = int((phone.end - offset_sec) * sr)   # within grain

        entry_lo = max(0, onset_samp - half_frame)
        entry_hi = min(len(raw_grain), onset_samp + half_frame)
        mfcc_entry = compute_mfcc(raw_grain[entry_lo:entry_hi], sr).tolist()

        exit_lo = max(0, phone_end_samp - half_frame)
        exit_hi = min(len(raw_grain), phone_end_samp + half_frame)
        mfcc_exit = compute_mfcc(raw_grain[exit_lo:exit_hi], sr).tolist()

        grain = hann_fade(raw_grain, fade_samples)

        # write grain file
        phoneme_dir = os.path.join(OUT_DIR, label)
        idx         = counts.get(label, 0)
        grain_path  = os.path.join(phoneme_dir, f"{idx:04d}.wav")
        write_wav_mono(grain_path, grain, sr)
        counts[label] = idx + 1

        index.append({
            "phoneme":       label,
            "prev_phone":    prev_label,
            "next_phone":    next_label,
            "word":          word_label,
            "path":          grain_path,
            "offset":        offset_sec,
            "cutoff":        cutoff_sec,
            "preutterance":  PREUTTERANCE,
            "overlap":       OVERLAP,
            "duration":      duration,
            "source_start":  phone.start,
            "source_end":    phone.end,
            "mfcc_entry":    mfcc_entry,
            "mfcc_exit":     mfcc_exit,
        })

    # -- write index ---------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    index_path = os.path.join(OUT_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nExtracted {len(index)} grains across {len(counts)} phoneme classes.")
    print(f"Index written to {index_path}")
    for ph, n in sorted(counts.items()):
        print(f"  {ph:6s}  {n:3d} grains")


if __name__ == "__main__":
    main()
