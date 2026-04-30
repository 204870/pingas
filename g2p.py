#!/usr/bin/env python3
"""
g2p.py

Grapheme-to-phoneme conversion: raw orthographic text → flat ARPAbet phone
sequence for synth.py.

Lookup strategy (in order of preference):
  1. Dictionary match from english_us_arpa.dict  (pure Python, ARM-safe)
  2. MFA PyniniGenerator for OOV words            (requires pynini + MFA,
                                                   no kalpy-kaldi needed)

The MFA import is lazy so the dict-only path works on machines where MFA's
full stack (kalpy-kaldi) is unavailable — e.g. ARM Linux.  Attempting to
synthesise a word that is both OOV and has no G2P backend will raise a
RuntimeError listing the missing words.

CLI usage:
  python g2p.py "hello world"
"""

import re
import sys

DICT_PATH  = "english_us_arpa.dict"
G2P_MODEL  = "english_us_arpa"


# ---------------------------------------------------------------------------
# Dictionary loading
# ---------------------------------------------------------------------------

def load_dict(path: str = DICT_PATH) -> dict[str, list[str]]:
    """
    Parse the MFA extended pronunciation dictionary into {word: [phones]}.

    The dict uses two tab-delimited layouts:
      word \\t phone1 phone2 ...
      word \\t prob \\t sil_before \\t sil_after \\t nsil_prob \\t phone1 phone2 ...

    When a word has multiple pronunciations the one with the highest
    pronunciation probability (column 2) is kept, matching the behaviour of
    MFA's aligner when it picks a single canonical form.
    """
    best: dict[str, tuple[float, list[str]]] = {}

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue

            word = cols[0].lower()

            # skip non-lexical tokens such as !SIL, <eps>, [bracketed]
            if not re.fullmatch(r"[a-z''\-]+", word):
                continue

            phones = cols[-1].split()

            # col[1] is the pronunciation probability when there are >2 cols
            # and the value is numeric; otherwise treat as probability 1.0
            prob = 1.0
            if len(cols) > 2:
                try:
                    prob = float(cols[1])
                except ValueError:
                    pass

            if word not in best or prob > best[word][0]:
                best[word] = (prob, phones)

    return {w: ph for w, (_, ph) in best.items()}


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Split text into lowercase word tokens.

    Keeps apostrophes and hyphens so contractions ("don't") and hyphenated
    compounds ("well-known") survive; they are present as entries in the dict.
    Digits and punctuation are dropped.
    """
    return re.findall(r"[a-z][a-z'\-]*", text.lower())


# ---------------------------------------------------------------------------
# OOV G2P via MFA PyniniGenerator
# ---------------------------------------------------------------------------

def _g2p_oov(words: list[str], model: str = G2P_MODEL) -> dict[str, list[str]]:
    """
    Generate pronunciations for OOV words using MFA's PyniniGenerator.

    Requires montreal-forced-aligner (and therefore pynini) but NOT
    kalpy-kaldi, so it runs on x86 Linux/macOS even when kaldi wheels are
    unavailable for the host architecture.

    Returns {word: [phones]}.  Words for which the model produces no output
    are silently omitted; the caller handles them as hard OOVs.
    """
    try:
        from montreal_forced_aligner.g2p.generator import PyniniGenerator
    except ImportError as exc:
        raise RuntimeError(
            "MFA PyniniGenerator is unavailable on this platform.\n"
            "Install montreal-forced-aligner on an x86 machine to handle "
            f"OOV words: {words}"
        ) from exc

    gen = PyniniGenerator(
        g2p_model_path=model,
        num_pronunciations=1,
    )
    gen.setup()
    raw: dict = gen.generate_pronunciations(words)
    gen.cleanup()

    out: dict[str, list[str]] = {}
    for word, prons in raw.items():
        if not prons:
            continue
        # prons may be [[phone, ...], ...] or [Pronunciation(...), ...]
        first = prons[0]
        if isinstance(first, str):
            # already a flat phone string
            out[word] = first.split()
        elif hasattr(first, "phones"):
            out[word] = list(first.phones)
        elif isinstance(first, (list, tuple)):
            out[word] = [str(p) for p in first]
        else:
            out[word] = str(first).split()

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def text_to_phones(
    text: str,
    dict_path: str = DICT_PATH,
    g2p_model: str = G2P_MODEL,
) -> list[str]:
    """
    Convert raw orthographic text to a flat ARPAbet phone sequence.

    All dictionary words are resolved in pure Python.  OOV words are batched
    through MFA's PyniniGenerator in a single pass so FST setup cost is paid
    once regardless of how many OOVs appear.

    Raises RuntimeError if any words remain unresolvable after G2P.
    """
    words   = tokenize(text)
    lexicon = load_dict(dict_path)

    resolved: dict[str, list[str]] = {}
    oovs:     list[str]            = []

    for w in words:
        if w in lexicon:
            resolved[w] = lexicon[w]
        elif w not in resolved:
            oovs.append(w)

    if oovs:
        print(f"  [g2p] {len(oovs)} OOV(s), running model: {oovs}")
        generated = _g2p_oov(oovs, g2p_model)
        resolved.update(generated)

        still_missing = [w for w in oovs if w not in resolved]
        if still_missing:
            raise RuntimeError(
                f"No pronunciation found for: {still_missing}\n"
                "Add them to the dictionary or use a different G2P model."
            )

    flat: list[str] = []
    for w in words:
        flat.extend(resolved[w])
    return flat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "hello world"
    phones = text_to_phones(text)
    print(" ".join(phones))
