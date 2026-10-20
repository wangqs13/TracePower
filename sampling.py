"""Exact weak Schur sampling via RSK row insertion.

Equal symbols remain in a row; insertion bumps the first strictly greater symbol.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def prefix_shapes(word, dimension, sizes, retained):
    """Top retained rows of RSK(word[:n]) for each requested n.

    Rows below retained can be discarded: row insertion only moves downward.
    Trailing zero columns are dropped; no positive coordinate is removed.
    """
    maximum = len(word)
    capacity = min(dimension, maximum, retained)
    lengths = np.zeros(capacity, dtype=np.int32)
    offsets = np.zeros(capacity + 1, dtype=np.int64)
    for row in range(capacity):
        offsets[row + 1] = offsets[row] + maximum // (row + 1)
    entries = np.empty(offsets[-1], dtype=np.int64)
    snapshots = np.zeros((len(sizes), capacity), dtype=np.int32)
    point, used = 0, 0
    for position in range(maximum):
        bumped = word[position]
        for row in range(capacity):
            start = offsets[row]
            lo, hi = 0, lengths[row]
            while lo < hi:
                mid = (lo + hi) // 2
                if entries[start + mid] <= bumped:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == lengths[row]:
                entries[start + lo] = bumped
                lengths[row] += 1
                used = max(used, row + 1)
                break
            prior = entries[start + lo]
            entries[start + lo] = bumped
            bumped = prior
        if point < len(sizes) and position + 1 == sizes[point]:
            snapshots[point, :used] = lengths[:used]
            point += 1
    return snapshots[:, :used].copy()



@njit(cache=True)
def counted_prefix_shapes(word, dimension, sizes, retained):
    """Exact row insertion with symbol counts for long words on small alphabets.

    A sorted tableau row is represented by its multiplicities. Replacing its
    first strictly greater symbol is identical to the packed-row insertion
    above, but requires O(retained * dimension) storage independent of n.
    """
    capacity = min(dimension, len(word), retained)
    counts = np.zeros((capacity, dimension), dtype=np.int32)
    lengths = np.zeros(capacity, dtype=np.int32)
    last = np.full(capacity, -1, dtype=np.int64)
    snapshots = np.zeros((len(sizes), capacity), dtype=np.int32)
    point, used = counted_insert_chunk(word, sizes, counts, lengths, last, snapshots, 0, 0, 0)
    return snapshots[:, :used].copy()



@njit(cache=True)
def counted_insert_chunk(word, sizes, counts, lengths, last, snapshots, consumed, point, used):
    """Continue exact RSK insertion, retaining the tableau across word chunks."""
    capacity = len(lengths)
    for position in range(len(word)):
        bumped = word[position]
        for row in range(capacity):
            counts[row, bumped] += 1
            if bumped >= last[row]:
                lengths[row] += 1
                last[row] = bumped
                used = max(used, row + 1)
                break
            successor = bumped + 1
            while counts[row, successor] == 0:
                successor += 1
            counts[row, successor] -= 1
            if successor == last[row] and counts[row, successor] == 0:
                last[row] -= 1
                while counts[row, last[row]] == 0:
                    last[row] -= 1
            bumped = successor
        if point < len(sizes) and consumed + position + 1 == sizes[point]:
            snapshots[point, :used] = lengths[:used]
            point += 1
    return point, used



def streamed_prefix_shapes(rng, family, dimension, sizes, retained, chunk_size=2**20):
    """Exact prefix shapes with memory independent of the total word length.

    PCG64 uniform draws are consumed in the same order as a single large draw.
    Only the input word is chunked; the entire RSK tableau persists.
    """
    if chunk_size < 1:
        raise ValueError('Chunk size must be positive')
    maximum = int(sizes[-1])
    capacity = min(dimension, maximum, retained)
    dtype = np.int64 if maximum > np.iinfo(np.int32).max else np.int32
    counts = np.zeros((capacity, dimension), dtype=dtype)
    lengths = np.zeros(capacity, dtype=dtype)
    last = np.full(capacity, -1, dtype=np.int64)
    snapshots = np.zeros((len(sizes), capacity), dtype=dtype)
    point, used, consumed = 0, 0, 0
    while consumed < maximum:
        size = min(chunk_size, maximum - consumed)
        word = word_from_uniforms(rng.random(size), family, dimension)
        point, used = counted_insert_chunk(word, sizes, counts, lengths, last,
                                          snapshots, consumed, point, used)
        consumed += size
    assert point == len(sizes)
    return snapshots[:, :used].copy()



def target(family, d, q):
    if family == "uniform":
        return d ** (1 - q)
    if family == "head_flat":
        return 0.5 ** q * (1 + (d - 1) ** (1 - q))
    raise ValueError(f"Unknown family {family}")



def word_from_uniforms(uniforms, family, d):
    if family == "uniform":
        return np.floor(uniforms * d).astype(np.int64)
    if family == "head_flat":
        return np.where(uniforms < 0.5, 0, 1 + np.floor((uniforms - 0.5) * (2 * (d - 1)))).astype(np.int64)
    raise ValueError(f"Unknown family {family}")



def estimator_parameters(q, epsilon, d):
    """Return the retained row count m and number of independent groups k."""
    if q < 1:
        return d, 1
    epsilon_prime = (epsilon / 5) ** (1 / (q - 1)) if q < 2 else epsilon / (q + 3)
    reciprocal = 1 / epsilon_prime
    cutoff = round(reciprocal) if np.isclose(reciprocal, round(reciprocal), rtol=1e-13) else math.ceil(reciprocal)
    m = min(d, cutoff)
    k = math.ceil(math.log(6 * m))
    return m, k + 1 - k % 2

