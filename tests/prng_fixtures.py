from __future__ import annotations

import os
import random

import numpy as np


def generate_urandom(n: int) -> np.ndarray:
    """Generate n true random bits from os.urandom."""
    n_bytes = (n + 7) // 8
    raw = os.urandom(n_bytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n]
    return bits.astype(np.uint8)


def generate_mt(n: int, seed: int = 42) -> np.ndarray:
    """Generate n bits from Mersenne Twister (random.getrandbits)."""
    rng = random.Random(seed)
    n_words = (n + 31) // 32
    words = [rng.getrandbits(32) for _ in range(n_words)]
    arr = np.array(words, dtype=np.uint32)
    bits = np.unpackbits(arr.view(np.uint8))[:n]
    return bits.astype(np.uint8)


def generate_lfsr(n: int, seed: int = 0xACE12345) -> np.ndarray:
    """Generate n bits from 32-bit maximal length LFSR.

    Polynomial: x^32 + x^22 + x^2 + x^1 + 1 (taps at 31, 21, 1, 0).
    """
    state = seed if seed != 0 else 1
    bits = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        bits[i] = state & 1
        fb = ((state >> 0) ^ (state >> 1) ^ (state >> 21) ^ (state >> 31)) & 1
        state = (state >> 1) | (fb << 31)
    return bits
