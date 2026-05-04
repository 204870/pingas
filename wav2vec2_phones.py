#!/usr/bin/env python3
"""
Test wav2vec2 phoneme recognition with frame-level timestamps.
"""

import sys
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Model trained on phoneme recognition (IPA output via espeak)
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


def load_audio(path: str, target_sr: int = 16000):
    """Load audio and resample to target sample rate."""
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != target_sr:
        import resampy
        audio = resampy.resample(audio, sr, target_sr)
    return audio, target_sr


def recognize_phones(audio: np.ndarray, sr: int, processor, model, device):
    """Run CTC phoneme recognition, return frame-level predictions."""
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]

    # Decode frame by frame
    frames = pred_ids.cpu().numpy()
    return frames, logits.shape[1]


def frames_to_segments(frames, vocab, frame_duration: float):
    """Convert frame predictions to segments with timestamps, collapsing CTC blanks."""
    SKIP_TOKENS = {"<pad>", "<s>", "</s>", "|", " ", ""}

    segments = []
    current_phone = None
    start_frame = 0

    for i, frame_id in enumerate(frames):
        phone = vocab[frame_id]

        # Skip blank/special tokens - they extend the current phone
        if phone in SKIP_TOKENS:
            continue

        if phone != current_phone:
            # Close previous segment
            if current_phone is not None:
                segments.append({
                    "phone": current_phone,
                    "start": start_frame * frame_duration,
                    "end": i * frame_duration,
                })
            start_frame = i
            current_phone = phone

    # Final segment
    if current_phone is not None:
        segments.append({
            "phone": current_phone,
            "start": start_frame * frame_duration,
            "end": len(frames) * frame_duration,
        })

    return segments


# IPA (espeak) to ARPAbet mapping
IPA_TO_ARPABET = {
    # Vowels
    'i': 'IY', 'ɪ': 'IH', 'e': 'EY', 'ɛ': 'EH', 'æ': 'AE',
    'ɑ': 'AA', 'ɔ': 'AO', 'o': 'OW', 'ʊ': 'UH', 'u': 'UW',
    'ʌ': 'AH', 'ə': 'AH', 'ɚ': 'ER', 'ɝ': 'ER',
    'eɪ': 'EY', 'aɪ': 'AY', 'ɔɪ': 'OY', 'aʊ': 'AW', 'oʊ': 'OW',
    'iː': 'IY', 'uː': 'UW', 'ɑː': 'AA', 'ɔː': 'AO', 'ɜː': 'ER',
    # Consonants
    'p': 'P', 'b': 'B', 't': 'T', 'd': 'D', 'k': 'K', 'ɡ': 'G', 'g': 'G',
    'tʃ': 'CH', 'dʒ': 'JH', 'f': 'F', 'v': 'V', 'θ': 'TH', 'ð': 'DH',
    's': 'S', 'z': 'Z', 'ʃ': 'SH', 'ʒ': 'ZH', 'h': 'HH', 'ɹ': 'R', 'r': 'R',
    'm': 'M', 'n': 'N', 'ŋ': 'NG', 'l': 'L', 'w': 'W', 'j': 'Y',
    'ɾ': 'D',  # flap
    # Silence
    '': '', ' ': '', '|': '',
}


def ipa_to_arpabet(ipa: str) -> str:
    """Convert IPA symbol to ARPAbet. Returns original if no mapping."""
    # Try direct lookup
    if ipa in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[ipa]
    # Try digraphs first (longer sequences)
    for ipa_seq, arpa in sorted(IPA_TO_ARPABET.items(), key=lambda x: -len(x[0])):
        if ipa.startswith(ipa_seq) and ipa_seq:
            return arpa
    return ipa.upper()  # fallback: uppercase original


def write_textgrid(segments, duration: float, output_path: str, convert_to_arpabet: bool = True):
    """Write segments to Praat TextGrid using praatio."""
    from praatio import textgrid as tgio
    from praatio.utilities.constants import Interval

    # Convert to ARPAbet if requested
    if convert_to_arpabet:
        for seg in segments:
            seg["phone"] = ipa_to_arpabet(seg["phone"])
        segments = [s for s in segments if s["phone"]]

    # Build phone intervals
    phone_intervals = [
        Interval(seg["start"], seg["end"], seg["phone"])
        for seg in segments
    ]

    # Create TextGrid with words and phones tiers
    tg = tgio.Textgrid()

    words_tier = tgio.IntervalTier(
        "words",
        [Interval(0, duration, "")],
        minT=0,
        maxT=duration
    )
    phones_tier = tgio.IntervalTier(
        "phones",
        phone_intervals,
        minT=0,
        maxT=duration
    )

    tg.addTier(words_tier)
    tg.addTier(phones_tier)
    tg.save(output_path, format="short_textgrid", includeBlankSpaces=True)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <wav_file> [output.TextGrid]")
        sys.exit(1)

    wav_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else wav_path.rsplit(".", 1)[0] + "_w2v2.TextGrid"

    print(f"Loading model {MODEL_ID}...")
    device = "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
    model.eval()

    print(f"Loading audio {wav_path}...")
    audio, sr = load_audio(wav_path)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s, Sample rate: {sr}")

    print("Running phoneme recognition...")
    frames, n_frames = recognize_phones(audio, sr, processor, model, device)

    # wav2vec2 outputs at 50Hz (20ms per frame)
    frame_duration = duration / n_frames
    print(f"  {n_frames} frames, {frame_duration*1000:.1f}ms per frame")

    # Build vocab lookup
    vocab = {v: k for k, v in processor.tokenizer.get_vocab().items()}

    # Convert to segments
    segments = frames_to_segments(frames, vocab, frame_duration)
    print(f"  {len(segments)} phone segments")

    # Filter out blank/silence tokens (model-specific)
    segments = [s for s in segments if s["phone"] not in ("|", " ", "")]
    print(f"  {len(segments)} after filtering blanks")

    # Preview first 20 segments
    print("\nFirst 20 segments:")
    for seg in segments[:20]:
        print(f"  {seg['start']:7.3f} - {seg['end']:7.3f}  {seg['phone']}")

    # Write TextGrid
    write_textgrid(segments, duration, output_path)
    print(f"\nTextGrid saved to {output_path}")


if __name__ == "__main__":
    main()
