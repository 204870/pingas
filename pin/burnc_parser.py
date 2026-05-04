"""
BURNC corpus parser for prosodic feature extraction (phone-level).

Extracts Prominence and Boundary features aligned with F0 and intensity targets
at the phone level. Uses abstract phone features (is_vowel, is_voiced) that
generalize across TIMIT (BURNC) and ARPAbet (synth.py).
"""

import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Phone:
    """A phone with its prosodic annotations and acoustic targets."""
    label: str
    start: float
    end: float
    # Abstract phone features (derived from label)
    is_vowel: bool = False
    is_voiced: bool = False
    stressed: bool = False
    # Prosodic features (from ToBI annotations)
    prominent: bool = False
    boundary: int = 0  # 0=none, 1=intermediate, 2=intonational
    boundary_rising: bool = False  # True if H% (question), False if L% or no boundary
    phrase_position: float = 0.0  # 0.0-1.0 position within current phrase
    # Acoustic targets
    f0_mean: Optional[float] = None
    f0_max: Optional[float] = None
    intensity_mean: Optional[float] = None
    duration: Optional[float] = None


@dataclass
class Utterance:
    """A parsed BURNC utterance."""
    utterance_id: str
    speaker: str
    phones: list[Phone] = field(default_factory=list)


# TIMIT phone classifications
# Reference: https://catalog.ldc.upenn.edu/docs/LDC93S1/PHONCODE.TXT
TIMIT_VOWELS = {
    'IY', 'IH', 'EH', 'EY', 'AE', 'AA', 'AW', 'AY', 'AH', 'AO', 'OY', 'OW',
    'UH', 'UW', 'UX', 'ER', 'AX', 'IX', 'AXR', 'AX-H'
}

TIMIT_VOICED = TIMIT_VOWELS | {
    # Voiced stops
    'B', 'D', 'G',
    # Voiced fricatives
    'V', 'DH', 'Z', 'ZH',
    # Nasals
    'M', 'N', 'NG', 'EM', 'EN', 'ENG',
    # Liquids and glides
    'L', 'R', 'W', 'Y', 'EL',
    # Flap
    'DX',
    # Affricates (voiced)
    'JH',
}

# Phones to skip
SKIP_PHONES = {'H#', 'PAU', 'brth', 'BRTH', 'SIL', '', '#'}


def parse_timit_phone(label: str) -> tuple[str, bool, bool, bool]:
    """
    Parse a TIMIT phone label.

    Returns: (base_phone, is_vowel, is_voiced, stressed)
    """
    # Check for stress marker
    stressed = '+1' in label
    base = label.replace('+1', '').replace('+0', '')

    # Skip closure markers for voicing check (they're unvoiced silence)
    if base.endswith('CL'):
        return base, False, False, False

    is_vowel = base in TIMIT_VOWELS
    is_voiced = base in TIMIT_VOICED

    return base, is_vowel, is_voiced, stressed


def parse_label_file(path: Path) -> list[tuple[float, str]]:
    """Parse ESPS-style label files (.ton, .brk, .lba)."""
    labels = []
    in_data = False

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '#':
                in_data = True
                continue
            if not in_data or not line:
                continue

            parts = line.split()
            if len(parts) >= 3:
                time = float(parts[0])
                label = parts[-1]
                labels.append((time, label))

    return labels


def parse_f0a(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse F0 ASCII file. Returns (time, f0, voicing_prob, rms)."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def phones_from_lba(lba_labels: list[tuple[float, str]]) -> list[Phone]:
    """Convert .lba labels to Phone objects."""
    phones = []
    prev_end = 0.0

    for end_time, label in lba_labels:
        if label in SKIP_PHONES:
            prev_end = end_time
            continue

        base, is_vowel, is_voiced, stressed = parse_timit_phone(label)

        phones.append(Phone(
            label=label,
            start=prev_end,
            end=end_time,
            is_vowel=is_vowel,
            is_voiced=is_voiced,
            stressed=stressed,
            duration=end_time - prev_end
        ))
        prev_end = end_time

    return phones


def assign_prominence(phones: list[Phone], ton_labels: list[tuple[float, str]]) -> None:
    """Mark phones as prominent if a pitch accent (*) falls within their boundaries."""
    accent_times = [t for t, label in ton_labels if '*' in label]

    for phone in phones:
        for t in accent_times:
            if phone.start - 0.01 <= t <= phone.end + 0.01:
                phone.prominent = True
                break


def assign_boundaries(phones: list[Phone], brk_labels: list[tuple[float, str]], ton_labels: list[tuple[float, str]]) -> None:
    """
    Assign boundary strength and direction from break indices and ToBI tones.

    - boundary: 0=none, 1=intermediate (3), 2=intonational (4)
    - boundary_rising: True if boundary tone ends in H% (question/continuation)
    """
    # Extract boundary tones (end in %) and their direction
    # L-L%, L-H% end in L% or H%
    boundary_tones = {}
    for time, label in ton_labels:
        if '%' in label:
            # H% = rising, L% = falling
            is_rising = 'H%' in label
            boundary_tones[time] = is_rising

    for phone in phones:
        # Assign break index
        for brk_time, brk_label in brk_labels:
            if abs(brk_time - phone.end) < 0.05:
                try:
                    idx = int(brk_label.rstrip('-p'))
                    if idx >= 4:
                        phone.boundary = 2
                    elif idx >= 3:
                        phone.boundary = 1
                except ValueError:
                    pass
                break

        # If there's a boundary, check if it's rising
        if phone.boundary > 0:
            for tone_time, is_rising in boundary_tones.items():
                if abs(tone_time - phone.end) < 0.1:  # Slightly larger tolerance for tone alignment
                    phone.boundary_rising = is_rising
                    break


def assign_phrase_positions(phones: list[Phone]) -> None:
    """
    Assign phrase position (0.0-1.0) to each phone.

    Phrases are delimited by boundary >= 1 (intermediate or intonational).
    Position helps model declination and boundary anticipation.
    """
    if not phones:
        return

    # Find phrase boundaries
    phrase_starts = [0]
    for i, p in enumerate(phones):
        if p.boundary > 0 and i < len(phones) - 1:
            phrase_starts.append(i + 1)

    # Assign positions within each phrase
    for phrase_idx in range(len(phrase_starts)):
        start_idx = phrase_starts[phrase_idx]
        end_idx = phrase_starts[phrase_idx + 1] if phrase_idx + 1 < len(phrase_starts) else len(phones)

        phrase_len = end_idx - start_idx
        if phrase_len <= 1:
            # Single phone phrase
            if start_idx < len(phones):
                phones[start_idx].phrase_position = 1.0
        else:
            for i in range(start_idx, end_idx):
                phones[i].phrase_position = (i - start_idx) / (phrase_len - 1)


# F0 bounds for valid pitch (avoid unvoiced/artifacts)
F0_MIN_HZ = 50.0
F0_MAX_HZ = 600.0


def hz_to_cents(f0_hz: float, ref_hz: float) -> float:
    """
    Convert F0 in Hz to cents relative to a reference frequency.

    cents = 1200 * log2(f0 / ref)

    One semitone = 100 cents, one octave = 1200 cents.
    """
    if f0_hz <= 0 or ref_hz <= 0:
        return 0.0
    return 1200.0 * np.log2(f0_hz / ref_hz)


def cents_to_hz(cents: float, ref_hz: float) -> float:
    """Convert cents back to Hz given a reference frequency."""
    return ref_hz * (2.0 ** (cents / 1200.0))


def assign_acoustic_targets(
    phones: list[Phone],
    time: np.ndarray,
    f0: np.ndarray,
    voicing: np.ndarray,
    rms: np.ndarray,
    voicing_threshold: float = 0.5
) -> None:
    """Assign F0 and intensity targets to each phone."""
    for phone in phones:
        mask = (time >= phone.start) & (time <= phone.end)
        if not np.any(mask):
            continue

        phone.intensity_mean = float(np.mean(rms[mask]))

        # F0 only for voiced frames within valid range
        phone_f0 = f0[mask]
        phone_voicing = voicing[mask]

        valid_mask = (
            (phone_voicing > voicing_threshold) &
            (phone_f0 >= F0_MIN_HZ) &
            (phone_f0 <= F0_MAX_HZ)
        )

        if np.any(valid_mask):
            valid_f0 = phone_f0[valid_mask]
            phone.f0_mean = float(np.mean(valid_f0))
            phone.f0_max = float(np.max(valid_f0))


def parse_utterance(base_path: Path) -> Optional[Utterance]:
    """Parse a single BURNC utterance."""
    lba_path = base_path.with_suffix('.lba')
    ton_path = base_path.with_suffix('.ton')
    brk_path = base_path.with_suffix('.brk')
    f0a_path = base_path.with_suffix('.f0a')

    for p in [lba_path, ton_path, brk_path, f0a_path]:
        if not p.exists():
            return None

    lba_labels = parse_label_file(lba_path)
    ton_labels = parse_label_file(ton_path)
    brk_labels = parse_label_file(brk_path)
    time, f0, voicing, rms = parse_f0a(f0a_path)

    phones = phones_from_lba(lba_labels)
    if not phones:
        return None

    assign_prominence(phones, ton_labels)
    assign_boundaries(phones, brk_labels, ton_labels)
    assign_phrase_positions(phones)
    assign_acoustic_targets(phones, time, f0, voicing, rms)

    utterance_id = base_path.name
    speaker = utterance_id[:3]

    return Utterance(utterance_id=utterance_id, speaker=speaker, phones=phones)


def find_utterances(burnc_root: Path) -> list[Path]:
    """Find all utterance base paths with complete annotations."""
    utterances = set()

    for lba_file in burnc_root.rglob('*.lba'):
        base = lba_file.with_suffix('')
        if (base.with_suffix('.ton').exists() and
            base.with_suffix('.brk').exists() and
            base.with_suffix('.f0a').exists()):
            utterances.add(base)

    return sorted(utterances)


def parse_corpus(burnc_root: Path, verbose: bool = False) -> list[Utterance]:
    """Parse the entire BURNC corpus."""
    utterance_paths = find_utterances(burnc_root)

    if verbose:
        print(f"Found {len(utterance_paths)} utterances with complete annotations")

    utterances = []
    for i, base_path in enumerate(utterance_paths):
        utt = parse_utterance(base_path)
        if utt is not None:
            utterances.append(utt)

        if verbose and (i + 1) % 100 == 0:
            print(f"  Parsed {i + 1}/{len(utterance_paths)}")

    if verbose:
        print(f"Successfully parsed {len(utterances)} utterances")

    return utterances


def utterances_to_sequences(
    utterances: list[Utterance],
    sequence_length: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Convert utterances to padded sequences for LSTM training.

    Input features (per phone):
      - is_vowel (0/1)
      - is_voiced (0/1)
      - stressed (0/1)
      - prominent (0/1) — pitch accent, maps to * in input
      - boundary (0/1) — phrase boundary present
      - boundary_rising (0/1) — H% vs L%, maps to ? vs . in input
      - phrase_position (0-1) — position within phrase, helps with declination

    Output targets (per phone):
      - f0_delta_cents: F0 change from previous phone in cents
      - intensity_delta: intensity change from previous phone (normalized)

    Returns: (X, y, masks, meta)
    """
    N_FEATURES = 7

    # Collect stats for normalization
    all_f0_deltas: list[float] = []
    all_int_deltas: list[float] = []

    # First pass: compute deltas and collect stats
    for utt in utterances:
        prev_f0 = None
        prev_int = None
        for p in utt.phones:
            if p.f0_mean is not None and prev_f0 is not None:
                # Delta in cents between consecutive phones
                delta = hz_to_cents(p.f0_mean, prev_f0)
                all_f0_deltas.append(delta)
            if p.f0_mean is not None:
                prev_f0 = p.f0_mean

            if p.intensity_mean is not None and prev_int is not None:
                all_int_deltas.append(p.intensity_mean - prev_int)
            if p.intensity_mean is not None:
                prev_int = p.intensity_mean

    # Stats for normalization (intensity delta only, F0 delta stays in cents)
    int_delta_std = np.std(all_int_deltas) + 1e-6 if all_int_deltas else 1.0

    # F0 delta stats for reference
    f0_delta_std = np.std(all_f0_deltas) if all_f0_deltas else 100.0

    if sequence_length is None:
        sequence_length = max(len(utt.phones) for utt in utterances)

    X_list, y_list, mask_list = [], [], []

    for utt in utterances:
        X_seq = np.zeros((sequence_length, N_FEATURES), dtype=np.float32)
        y_seq = np.zeros((sequence_length, 2), dtype=np.float32)
        mask = np.zeros(sequence_length, dtype=np.float32)

        prev_f0 = None
        prev_int = None

        for i, p in enumerate(utt.phones[:sequence_length]):
            X_seq[i] = [
                float(p.is_vowel),
                float(p.is_voiced),
                float(p.stressed),
                float(p.prominent),
                float(p.boundary > 0),  # binary: is there a boundary?
                float(p.boundary_rising),  # is it rising (?) vs falling (.)
                p.phrase_position  # 0-1 position in phrase
            ]

            # Compute deltas
            if p.f0_mean is not None and p.intensity_mean is not None:
                if prev_f0 is not None and prev_int is not None:
                    # F0 delta in cents (how much pitch changed from previous phone)
                    f0_delta = hz_to_cents(p.f0_mean, prev_f0)
                    # Intensity delta normalized
                    int_delta = (p.intensity_mean - prev_int) / int_delta_std

                    y_seq[i] = [f0_delta, int_delta]
                    mask[i] = 1.0

                prev_f0 = p.f0_mean
                prev_int = p.intensity_mean

        X_list.append(X_seq)
        y_list.append(y_seq)
        mask_list.append(mask)

    meta = {
        'f0_delta_stats': {'std_cents': float(f0_delta_std)},
        'intensity_delta_stats': {'std': float(int_delta_std)},
        'sequence_length': sequence_length,
        'feature_names': [
            'is_vowel', 'is_voiced', 'stressed', 'prominent',
            'boundary', 'boundary_rising', 'phrase_position'
        ],
        'target_names': ['f0_delta_cents', 'intensity_delta_norm'],
        'f0_unit': 'cents (delta from previous phone)',
        'f0_bounds_hz': {'min': F0_MIN_HZ, 'max': F0_MAX_HZ}
    }

    return np.stack(X_list), np.stack(y_list), np.stack(mask_list), meta


# ARPAbet equivalents for inference in synth.py
ARPABET_VOWELS = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
    'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
}

ARPABET_VOICED = ARPABET_VOWELS | {
    'B', 'D', 'G',  # voiced stops
    'V', 'DH', 'Z', 'ZH',  # voiced fricatives
    'M', 'N', 'NG',  # nasals
    'L', 'R', 'W', 'Y',  # liquids/glides
    'JH',  # voiced affricate
}


def parse_arpabet_phone(label: str) -> tuple[bool, bool, bool]:
    """
    Parse an ARPAbet phone label (as used by MFA/synth.py).

    Returns: (is_vowel, is_voiced, stressed)

    Example:
        'EY1' -> (True, True, True)
        'K' -> (False, False, False)
        'M' -> (False, True, False)
    """
    # Strip stress marker to get base phone
    base = label.rstrip('012')

    # Stress is indicated by 1 or 2 suffix on vowels
    stressed = label[-1] in ('1', '2') if label else False

    is_vowel = base in ARPABET_VOWELS
    is_voiced = base in ARPABET_VOICED

    return is_vowel, is_voiced, stressed


def phones_to_features(
    phones: list[str],
    prominent: list[bool],
    boundary: list[bool],
    boundary_rising: list[bool],
    phrase_positions: Optional[list[float]] = None
) -> np.ndarray:
    """
    Convert ARPAbet phone sequence to feature matrix for inference.

    Args:
        phones: List of ARPAbet phone labels (e.g., ['DH', 'AH0', 'K', 'AE1', 'T'])
        prominent: Per-phone prominence flags (maps from *)
        boundary: Per-phone boundary flags (maps from . ? !)
        boundary_rising: Per-phone rising boundary flag (True for ?, False for . !)
        phrase_positions: Optional per-phone phrase positions (0-1). If None, computed automatically.

    Returns:
        (n_phones, 7) feature matrix
    """
    n = len(phones)
    X = np.zeros((n, 7), dtype=np.float32)

    # Compute phrase positions if not provided
    if phrase_positions is None:
        phrase_positions = []
        phrase_start = 0
        for i in range(n):
            if boundary[i] and i < n - 1:
                # End of phrase, compute positions for this phrase
                phrase_len = i - phrase_start + 1
                for j in range(phrase_start, i + 1):
                    pos = (j - phrase_start) / max(phrase_len - 1, 1)
                    phrase_positions.append(pos)
                phrase_start = i + 1
        # Handle remaining phones in last phrase
        if phrase_start < n:
            phrase_len = n - phrase_start
            for j in range(phrase_start, n):
                pos = (j - phrase_start) / max(phrase_len - 1, 1)
                phrase_positions.append(pos)

    for i, phone in enumerate(phones):
        is_vowel, is_voiced, stressed = parse_arpabet_phone(phone)
        X[i] = [
            float(is_vowel),
            float(is_voiced),
            float(stressed),
            float(prominent[i]),
            float(boundary[i]),
            float(boundary_rising[i]),
            phrase_positions[i] if i < len(phrase_positions) else 0.0
        ]

    return X


if __name__ == '__main__':
    import sys

    script_dir = Path(__file__).parent
    default_burnc = script_dir.parent.parent / 'bu_radio' / 'data'
    burnc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_burnc

    print(f"Parsing BURNC corpus at: {burnc_path}")
    utterances = parse_corpus(burnc_path, verbose=True)

    if utterances:
        print("\nSample utterance:")
        sample = utterances[0]
        print(f"  ID: {sample.utterance_id}, Speaker: {sample.speaker}")
        print(f"  Phones: {len(sample.phones)}")
        print(f"  {'Label':<8} {'Time':<15} {'V':>2} {'Vd':>3} {'St':>3} {'Pr':>3} {'Bd':>3} {'R':>2} {'Pos':>5} {'F0':>8}")
        for p in sample.phones[:15]:
            f0_str = f"{p.f0_mean:.1f}" if p.f0_mean else "  -"
            print(f"  {p.label:<8} [{p.start:.3f}-{p.end:.3f}] "
                  f"{int(p.is_vowel):>2} {int(p.is_voiced):>3} {int(p.stressed):>3} "
                  f"{int(p.prominent):>3} {p.boundary:>3} {int(p.boundary_rising):>2} "
                  f"{p.phrase_position:>5.2f} {f0_str:>8}")

    print("\n--- Converting to sequences ---")
    X, y, masks, meta = utterances_to_sequences(utterances)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"masks shape: {masks.shape}")

    print(f"\nF0 delta stats: std = {meta['f0_delta_stats']['std_cents']:.1f} cents")

    # Feature distribution
    print(f"\nFeature distributions (across all phones with targets):")
    all_X = X[masks.astype(bool)]
    for i, name in enumerate(meta['feature_names']):
        if name == 'phrase_position':
            print(f"  {name}: mean={np.mean(all_X[:, i]):.2f}")
        else:
            pct = 100 * np.mean(all_X[:, i] > 0)
            print(f"  {name}: {pct:.1f}% positive")

    # Test ARPAbet conversion (simulating "the *cat." input)
    print("\n--- ARPAbet feature extraction test ---")
    print("Simulating: 'the *cat.'")
    test_phones = ['DH', 'AH0', 'K', 'AE1', 'T']
    test_prom = [False, False, False, True, False]  # * on 'cat'
    test_bnd = [False, False, False, False, True]   # . after 'cat'
    test_rising = [False, False, False, False, False]  # . = falling
    test_X = phones_to_features(test_phones, test_prom, test_bnd, test_rising)
    print(f"Phones: {test_phones}")
    print(f"Features (vowel, voiced, stress, prom, bnd, rising, pos):\n{test_X}")
