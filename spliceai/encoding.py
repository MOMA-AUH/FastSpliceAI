import numpy as np

__all__ = ["one_hot_encode"]

_ONE_HOT_ENCODING = np.zeros((256, 4), dtype=np.float32)
_ONE_HOT_ENCODING[[ord(base) for base in "Aa"], 0] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Cc"], 1] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Gg"], 2] = 1
_ONE_HOT_ENCODING[[ord(base) for base in "Tt"], 3] = 1


def one_hot_encode(sequence: str) -> np.ndarray:
    """Encode a nucleotide sequence as an ``(length, 4)`` float array."""
    ascii_sequence = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return _ONE_HOT_ENCODING[ascii_sequence]
