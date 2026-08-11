"""Shared lightweight helpers."""

import re
from functools import cache
from typing import Any

import numpy as np

__all__ = [
    "is_valid_allele",
    "normalise_chrom",
    "one_hot_encode",
]

_ONE_HOT_ENCODING = np.zeros((256, 4), dtype=np.float32)
_ONE_HOT_ENCODING[[ord(base) for base in "Aa"], 0] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Cc"], 1] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Gg"], 2] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Tt"], 3] = 1

_IUPAC_DNA_ALLELE = re.compile(r"[ACGTRYSWKMBDHVN]+", re.IGNORECASE)


def one_hot_encode(sequence: str) -> np.ndarray:
    """Encode a nucleotide sequence as an ``(length, 4)`` float array."""
    ascii_sequence = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return _ONE_HOT_ENCODING[ascii_sequence]


@cache
def is_valid_allele(alt: Any) -> bool:
    """Return whether an ALT is a nonempty IUPAC DNA base string."""
    return isinstance(alt, str) and _IUPAC_DNA_ALLELE.fullmatch(alt) is not None


def normalise_chrom(source: str, target: str) -> str:
    """Match the presence of a ``chr`` prefix in a target chromosome name."""
    source_has_prefix = source.startswith("chr")
    target_has_prefix = target.startswith("chr")

    if source_has_prefix and not target_has_prefix:
        return source[3:]
    if not source_has_prefix and target_has_prefix:
        return "chr" + source
    return source
