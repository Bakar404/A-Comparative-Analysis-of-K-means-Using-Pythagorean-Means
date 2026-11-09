"""Gap statistic with an axis-aligned uniform null reference distribution."""

from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_array

from .kmeans import KMeans


def within_cluster_dispersion(X, labels):
    """Pairwise squared dispersion: sum_r sum_{i,j in r} ||xi-xj||^2 / (2*n_r).

    This equals SSE about each cluster's arithmetic center, independently of
    which centroid update rule produced the partition.
    """
    X = check_array(X, dtype=float)
    labels = np.asarray(labels)
    if labels.shape != (len(X),):
        raise ValueError("labels must contain one label per sample")
    return float(sum(
        ((members - members.mean(axis=0)) ** 2).sum()
        for label in np.unique(labels)
        for members in [X[labels == label]]
    ))


@dataclass(frozen=True)
class GapResult:
    k_values: np.ndarray
    gaps: np.ndarray
    standard_errors: np.ndarray
    dispersions: np.ndarray
    optimal_k: int
    selected_at_boundary: bool
    selections: dict
    empty_cluster_retries: list


def select_k(k_values, gaps, standard_errors, selection):
    """Return (K, boundary flag) from an already computed gap table.

    Dudoit & Fridlyand (2002), doi:10.1186/gb-2002-3-7-research0036:
    choose the smallest K within s at the global gap maximum.
    """
    if selection in ("max", "max_gap"):
        index = int(np.argmax(gaps))
        boundary = index == len(k_values) - 1
    elif selection == "dudoit_fridlyand":
        maximum = int(np.argmax(gaps))
        index = int(np.flatnonzero(gaps >= gaps[maximum] - standard_errors[maximum])[0])
        boundary = maximum == len(k_values) - 1
    elif selection == "one_se":
        candidates = np.flatnonzero(gaps[:-1] >= gaps[1:] - standard_errors[1:])
        index = int(candidates[0]) if len(candidates) else len(k_values) - 1
        boundary = not bool(len(candidates))
    else:
        raise ValueError("selection must be 'one_se', 'max', or 'dudoit_fridlyand'")
    return int(k_values[index]), boundary


def fit_nonempty(X, k, *, mean, rng, n_init=10, max_iter=300, max_retries=5):
    """Retry incomplete partitions with new seeds; return model and retry records.

    Exhaustion raises rather than reporting fewer than K occupied clusters.
    """
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, (int, np.integer))
        or max_retries < 0
    ):
        raise ValueError("max_retries must be a non-negative integer")
    retries = []
    for attempt in range(max_retries + 1):
        seed = int(rng.integers(2**32))
        if retries:
            retries[-1]["retry_seed"] = seed
        model = KMeans(
            k, mean=mean, random_state=seed, n_init=n_init, max_iter=max_iter,
        ).fit(X)
        occupied = len(np.unique(model.labels_))
        if occupied == k:
            return model, retries
        retries.append({"attempt": attempt, "seed": seed, "occupied_clusters": occupied})
    raise RuntimeError(
        f"Could not fit {k} occupied clusters after {max_retries + 1} attempts: {retries}"
    )


def gap_statistic(
    X, k_values=range(1, 21), *, mean="arithmetic", n_references=10,
    random_state=0, n_init=10, max_iter=300, selection="one_se", max_retries=5,
):
    """Select K by one-SE, maximum gap, or Dudoit & Fridlyand's global one-SE.

    ``one_se`` selects the first k with Gap(k) >= Gap(k+1) - s(k+1).
    If none qualifies it returns the largest k and flags the search boundary.
    ``max`` (also ``max_gap`` for compatibility) selects the smallest k attaining
    the maximum gap, flagging a
    maximum at the upper search boundary. Reference samples are shared across
    k. Use identical seeds across means for a paired comparison. Standard errors
    use sqrt(1 + 1/B) times the population standard deviation (ddof=0, divisor B),
    as in Tibshirani, Walther & Hastie (2001), doi:10.1111/1467-9868.00293.
    Empty-cluster fits are retried with new seeds up to ``max_retries`` times,
    recorded in the result, and raise on exhaustion.
    """
    X = check_array(X, dtype=float)
    if selection not in ("one_se", "max", "max_gap", "dudoit_fridlyand"):
        raise ValueError("selection must be 'one_se', 'max', or 'dudoit_fridlyand'")
    values = list(k_values)
    if (
        len(values) < 2
        or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in values)
        or values[0] < 1 or values[-1] >= len(X)
        or any(right != left + 1 for left, right in zip(values, values[1:]))
    ):
        raise ValueError("k_values must be consecutive positive integers below sample count")
    if (
        isinstance(n_references, bool)
        or not isinstance(n_references, (int, np.integer))
        or n_references < 2
    ):
        raise ValueError("n_references must be an integer of at least 2")
    rng = np.random.default_rng(random_state)
    references = rng.uniform(
        X.min(axis=0), X.max(axis=0), size=(n_references, *X.shape),
    )
    gaps, errors, dispersions, retries = [], [], [], []
    for k in values:
        log_reference = []
        for data_index, data in enumerate([X, *references]):
            model, fit_retries = fit_nonempty(
                data, k, mean=mean, rng=rng, n_init=n_init,
                max_iter=max_iter, max_retries=max_retries,
            )
            retries.extend(
                {"k": k, "data_index": data_index, **record} for record in fit_retries
            )
            dispersion = within_cluster_dispersion(data, model.labels_)
            if dispersion <= 0:
                raise ValueError("Gap statistic requires positive dispersion; reduce k")
            if data is X:
                observed = dispersion
            else:
                log_reference.append(np.log(dispersion))
        gaps.append(float(np.mean(log_reference) - np.log(observed)))
        errors.append(float(np.std(log_reference, ddof=0) * np.sqrt(1 + 1 / n_references)))
        dispersions.append(observed)
    values, gaps, errors = np.array(values), np.array(gaps), np.array(errors)
    selections = {
        rule: select_k(values, gaps, errors, rule)[0]
        for rule in ("dudoit_fridlyand", "max", "one_se")
    }
    optimal_k, boundary = select_k(values, gaps, errors, selection)
    return GapResult(
        values, gaps, errors, np.array(dispersions), optimal_k, boundary,
        selections, retries,
    )
