"""
pac_vc.py

Utilities to estimate an empirical VC-dimension (via shattering tests) and compute PAC
sample complexity bounds (VC-based and finite-hypothesis CLT-style bounds).

WARNING:
- Empirical VC by shattering is combinatorial and expensive. Use small `max_k` (<=6)
  and a small `max_train_samples` (e.g., 50-200) for practical runtimes.
- The PAC formulas are standard bounds; there are multiple variants in the literature.
  We provide commonly-used simplified versions.

Functions:
- empirical_vc_shatter(X_sym, max_k=6, max_train_samples=200, random_state=42)
- pac_sample_size_vc(d_vc, epsilon, delta)
- pac_sample_size_finite(log2_H, epsilon, delta)
- approximate_log2_hypotheses_from_features(n_features, max_depth)
"""

import itertools
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from typing import Tuple, Optional, List

def empirical_vc_shatter(X_sym: np.ndarray,
                         max_k: int = 6,
                         max_train_samples: int = 200,
                         classifier_ctor=DecisionTreeClassifier,
                         random_state: int = 42,
                         verbose: bool = True) -> Tuple[int, Optional[List[int]]]:
    """
    Empirically estimate the VC-dimension by brute-force shattering test on binary features.

    Parameters
    ----------
    X_sym : np.ndarray
        Binary symptom matrix (shape: n_samples x n_symptoms). Only binary columns expected.
    max_k : int
        Maximum subset size to attempt. Increase to try larger feature sets (exponential cost).
    max_train_samples : int
        If dataset is larger, will randomly subsample to at most this many rows (shattering tests use rows).
    classifier_ctor : callable
        A classifier constructor to attempt to realize labelings. We use DecisionTreeClassifier by default.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Print progress info.

    Returns
    -------
    (vc_empirical, subset_indices)
        vc_empirical : largest k found that is shattered (0 if none)
        subset_indices : list of feature indices for a shattered subset (one example), or None
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_sym.shape

    # Subsample rows to keep tests feasible
    if n_samples > max_train_samples:
        rows = rng.choice(n_samples, size=max_train_samples, replace=False)
        X = X_sym[rows]
        if verbose:
            print(f"[empirical_vc_shatter] subsampled {n_samples} -> {len(rows)} rows for tests")
    else:
        X = X_sym.copy()

    # ensure binary
    unique_vals = np.unique(X)
    if not set(unique_vals).issubset({0, 1}):
        if verbose:
            print("[empirical_vc_shatter] Warning: X contains non-binary values. Thresholding at >0.")
        X = (X != 0).astype(int)

    # We'll test from largest k down to 1 for early exit when found
    for k in range(min(max_k, n_features), 0, -1):
        if verbose:
            print(f"[empirical_vc_shatter] Trying subsets of size k={k} ...")
        # iterate over feature index subsets (combinatorial)
        for subset in itertools.combinations(range(n_features), k):
            X_sub = X[:, subset]  # shape: n_rows x k
            # For shattering, ALL 2^k labelings of the rows restricted to the columns must be realizable.
            # But the formal def of shattering: for the set of k points (here: columns?) Actually: we treat the k features
            # as the coordinate axes and the set of points are n_rows. The usual test for VC on features: we check sets
            # of k examples; but here we try feature-subset shattering in terms of label patterns across rows.
            # Practical approach used here: Test whether for the k selected FEATURE COLUMNS, we can realize ALL 2^k
            # labelings on the 2^k distinct *feature patterns* present among rows. We'll do a simpler but practical test:
            # Sample up to k samples (rows) with distinct patterns and test all labelings on those rows.
            #
            # Implementation: pick up to k rows with distinct feature-vectors; use those as the 'points to be shattered'.
            patterns, inverse = np.unique(X_sub, axis=0, return_inverse=True)
            if patterns.shape[0] < k:
                # Not enough distinct points to even form a k-point set; skip
                continue
            # select k distinct patterns (take the first k)
            selected_patterns = patterns[:k]
            # find row indices corresponding to each selected pattern (choose one row per pattern)
            row_indices = []
            for pat in selected_patterns:
                # find first row matching pattern
                for i, r in enumerate(X_sub):
                    if np.array_equal(r, pat):
                        row_indices.append(i)
                        break
            # Now we have k rows (row_indices). For shattering test, iterate ALL 2^k labelings of these k rows.
            shattered = True
            for label_bits in range(2 ** k):
                labels = [ (label_bits >> bit) & 1 for bit in range(k) ]
                # create full-label vector for the subsampled dataset: assign labels for rows in row_indices
                y = np.zeros(X_sub.shape[0], dtype=int)
                # For rows that share the same pattern, assign same label as the representative pattern index
                # find mapping from pattern index (inverse) to label
                pattern_label_map = {}
                for pidx, row_idx in enumerate(row_indices):
                    pat_idx = inverse[row_idx]
                    pattern_label_map[pat_idx] = labels[pidx]
                # assign labels by pattern
                for i_pat, lab in pattern_label_map.items():
                    y[inverse == i_pat] = lab
                # try to train a classifier to perfectly fit X_sub -> y
                clf = classifier_ctor()  # default params (for DT: can overfit)
                try:
                    clf.fit(X_sub, y)
                except Exception:
                    shattered = False
                    break
                preds = clf.predict(X_sub)
                if not np.array_equal(preds, y):
                    shattered = False
                    break
            if shattered:
                if verbose:
                    print(f"[empirical_vc_shatter] Found shattered subset of size {k}: indices={subset}")
                return k, list(subset)
        if verbose:
            print(f"[empirical_vc_shatter] No shattered subset of size {k}")
    return 0, None


def pac_sample_size_vc(d_vc: int, epsilon: float, delta: float) -> int:
    """
    Compute a PAC sample size bound using a common VC-bound approximation:

    m >= (1/epsilon) * ( ln(2/delta) + d_vc * ln(2*e/epsilon) )

    Parameters
    ----------
    d_vc : int
        VC dimension
    epsilon : float
        Desired error tolerance (0 < epsilon < 1)
    delta : float
        Confidence parameter (0 < delta < 1), we want probability >= 1 - delta

    Returns
    -------
    m : int
        Suggested minimum sample size
    """
    assert 0 < epsilon < 1
    assert 0 < delta < 1
    term = math.log(2.0 / delta) + d_vc * math.log((2.0 * math.e) / epsilon)
    m = (term / epsilon)
    return int(math.ceil(m))


def pac_sample_size_finite(log2_H: float, epsilon: float, delta: float) -> int:
    """
    PAC bound for finite hypothesis class H:

    m >= (1/epsilon) * ( ln(2/delta) + ln(|H|) )

    We accept log2_H (base-2 log) to avoid huge numbers: ln(|H|) = log2_H * ln(2)

    Parameters
    ----------
    log2_H : float
        log2(|H|)
    epsilon, delta : floats

    Returns
    -------
    m : int
    """
    assert 0 < epsilon < 1
    assert 0 < delta < 1
    ln_H = log2_H * math.log(2)
    term = math.log(2.0 / delta) + ln_H
    m = term / epsilon
    return int(math.ceil(m))


def approximate_log2_hypotheses_from_features(n_features: int, max_depth: int = 3) -> float:
    """
    Heuristic approx of log2(|H|) for decision trees on binary features.
    This is NOT rigorous; it's a quick heuristic for plugging into the finite-class bound.

    We approximate the number of different decision trees of depth <= D roughly by:
      sum_{t=1..T} (n_features)^{t}  (very rough). We then take log2.

    Parameters
    ----------
    n_features : int
    max_depth : int

    Returns
    -------
    approx_log2_H : float
    """
    total = 0.0
    # number of internal decision nodes up to depth D ~ 2^D - 1
    max_nodes = 2 ** max_depth - 1
    # very rough: each internal node selects a feature among n_features
    for t in range(1, max_nodes + 1):
        total += (n_features ** t)
    # to avoid huge numbers, take log2
    return math.log2(total + 1.0)


# ---------------- Example usage helper ----------------
def example_usage_with_dataset(X_full: np.ndarray, mlb=None):
    """
    Example: given the full feature matrix X from your build_features() (where X = [age, weight, symptoms...]),
    extract the binary symptom part and run empirical VC estimation and PAC computations.

    Assumes first two columns are age and weight, rest are binary symptom indicators.
    """
    if X_full.shape[1] <= 2:
        raise ValueError("X_full should contain binary symptom columns after first two numeric columns")
    X_sym = X_full[:, 2:].astype(int)
    # Estimate empirical VC dimension (use small k)
    vc_emp, subset = empirical_vc_shatter(X_sym, max_k=5, max_train_samples=150, verbose=True)
    print(f"Empirical VC (by shatter test, k<=5): {vc_emp}, example subset: {subset}")

    # PAC sample size for chosen eps, delta
    eps = 0.05
    delta = 0.05
    m_vc = pac_sample_size_vc(vc_emp if vc_emp>0 else 1, eps, delta)
    approx_log2H = approximate_log2_hypotheses_from_features(X_sym.shape[1], max_depth=3)
    m_finite = pac_sample_size_finite(approx_log2H, eps, delta)
    print(f"PAC sample size (VC-bound) for eps={eps}, delta={delta}, d_vc={vc_emp}: {m_vc}")
    print(f"PAC sample size (finite-hypothesis approx) using log2|H|~{approx_log2H:.2f}: {m_finite}")

    return {
        "vc_empirical": vc_emp,
        "vc_subset": subset,
        "pac_m_vc": m_vc,
        "pac_m_finite_approx": m_finite
    }
