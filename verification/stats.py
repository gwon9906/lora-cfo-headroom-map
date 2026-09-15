"""Interval estimates shared by the verification scripts.

Two regimes appear in this repo and they need different intervals:

  * i.i.d. symbol trials (detector comparisons, fixed-CFO sweeps)
      -> Wilson score interval on a binomial proportion.
  * packet-pooled trials (CFO estimated once per packet of `ns` symbols)
      -> symbol outcomes inside a packet share one CFO draw and one CFO
         estimate, so they are NOT independent.  The effective sample size is
         the number of PACKETS, not symbols.  We use a cluster bootstrap that
         resamples whole packets.
"""
import numpy as np

Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """95% Wilson score interval for k successes out of n, in percent."""
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p * 100, (c - h) * 100, (c + h) * 100


def fmt_wilson(k, n, w=0):
    p, lo, hi = wilson(k, n)
    return f"{p:{w}.1f}% [{lo:.1f}, {hi:.1f}]"


def cluster_boot(ok, ns, reps=10000, seed=0, z=Z95):
    """95% CI for the mean of `ok` (0/1 per symbol) when symbols are grouped
    into consecutive packets of size `ns` that share a nuisance parameter.
    Resamples packets with replacement.  Returns (mean%, lo%, hi%, n_packets).
    """
    ok = np.asarray(ok, dtype=np.float64)
    npk = len(ok) // ns
    g = ok[:npk * ns].reshape(npk, ns).mean(1)
    r = np.random.default_rng(seed)
    bs = g[r.integers(0, npk, size=(reps, npk))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return g.mean() * 100, lo * 100, hi * 100, npk


def paired_boot(ok_a, ok_b, ns, reps=10000, seed=0):
    """95% CI for the PAIRED difference (a - b) in percentage points, with the
    same packet clustering.  Both arms must be evaluated on the same packets.
    Returns (diff%p, lo, hi, p_two_sided_sign).
    """
    a = np.asarray(ok_a, float); b = np.asarray(ok_b, float)
    npk = len(a) // ns
    d = (a[:npk * ns].reshape(npk, ns) - b[:npk * ns].reshape(npk, ns)).mean(1)
    r = np.random.default_rng(seed)
    bs = d[r.integers(0, npk, size=(reps, npk))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return d.mean() * 100, lo * 100, hi * 100, p
