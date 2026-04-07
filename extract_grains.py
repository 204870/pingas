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

Grains are written to  grains/<PHONEME>/<index>.wav  (mono, 44100 Hz, int16).
A JSON index  grains/index.json  lists every grain with its metadata.
"""

import json
import wave
import struct
import os
import numpy as np
import tgt
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
    group      = tg[0]
    words_tier = group[0]   # "words" IntervalTier
    phones_tier = group[1]  # "phones" IntervalTier

    # -- also open with tgt for flat phone iteration -------------------------
    tg_flat = tgt.io.read_textgrid(TG_PATH)
    flat_phones = tg_flat.get_tier_by_name("phones").intervals  # list[Interval]

    # -- extract grains ------------------------------------------------------
    index   = []   # list of grain metadata dicts
    counts  = {}   # phoneme → number of grains written so far

    fade_samples = max(1, int(OVERLAP * sr))

    for interval in flat_phones:
        label = interval.text.strip()
        if not label:
            continue  # skip silence/empty

        duration = interval.end_time - interval.start_time
        if duration < MIN_DURATION:
            continue

        # OTO-style boundaries (clamped to audio extent)
        offset_sec  = max(0.0,       interval.start_time - PREUTTERANCE)
        cutoff_sec  = min(total_dur, interval.end_time   + OVERLAP)

        offset_samp = int(offset_sec * sr)
        cutoff_samp = int(cutoff_sec * sr)

        grain = audio[offset_samp:cutoff_samp]
        grain = hann_fade(grain, fade_samples)

        # write grain file
        phoneme_dir = os.path.join(OUT_DIR, label)
        idx         = counts.get(label, 0)
        grain_path  = os.path.join(phoneme_dir, f"{idx:04d}.wav")
        write_wav_mono(grain_path, grain, sr)
        counts[label] = idx + 1

        index.append({
            "phoneme":       label,
            "path":          grain_path,
            "offset":        offset_sec,
            "cutoff":        cutoff_sec,
            "preutterance":  PREUTTERANCE,
            "overlap":       OVERLAP,
            "duration":      duration,
            "source_start":  interval.start_time,
            "source_end":    interval.end_time,
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
