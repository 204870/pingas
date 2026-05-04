"""
PIN (Prosodic INtegration) module for prosody prediction.

Provides tools for training and applying prosody models based on
prominence and boundary annotations.

Input format for synthesis:
  - * before word = prominence (pitch accent)
  - . after word = falling boundary (declarative)
  - ? after word = rising boundary (question)
  - ! after word = emphatic (prominence + falling boundary)
"""

from .burnc_parser import (
    parse_corpus,
    utterances_to_sequences,
    phones_to_features,
    parse_arpabet_phone,
    hz_to_cents,
    cents_to_hz,
    F0_MIN_HZ,
    F0_MAX_HZ,
    Phone,
    Utterance,
)

__all__ = [
    'parse_corpus',
    'utterances_to_sequences',
    'phones_to_features',
    'parse_arpabet_phone',
    'hz_to_cents',
    'cents_to_hz',
    'F0_MIN_HZ',
    'F0_MAX_HZ',
    'Phone',
    'Utterance',
]
