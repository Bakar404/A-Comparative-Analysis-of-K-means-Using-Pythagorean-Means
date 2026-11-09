"""Euclidean assignments with interchangeable coordinate-wise centroid means."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted


MEANS = ("arithmetic", "geometric", "harmonic")


def centroid(points, mean="arithmetic"):
    """Compute a coordinate-wise mean; geometric/harmonic require positive data."""
    points = check_array(points, dtype=float)
    if mean not in MEANS:
        raise ValueError(f"mean must be one of {MEANS}")
    if mean == "arithmetic":
        return points.mean(axis=0)
    if np.any(points <= 0):
        raise ValueError("geometric and harmonic centroids require positive values")
    if mean == "geometric":
        return np.exp(np.log(points).mean(axis=0))
    return 1.0 / (1.0 / points).mean(axis=0)


class KMeans(ClusterMixin, BaseEstimator):
    """Seeded k-means++ initialization and Euclidean nearest-centroid assignments.

    Non-positive features are shifted to minimum 1 for geometric/harmonic means.
    Centers exposed to callers are translated back to the original coordinates.
    Empty centers are reseeded at farthest points. Multiple starts minimize squared
    distances to the variant's centers, not a guaranteed decreasing objective.
    """

    def __init__(
        self, n_clusters=3, *, mean="arithmetic", random_state=None,
        n_init=5, max_iter=300, tol=1e-4, init="k-means++",
    ):
        self.n_clusters = n_clusters
        self.mean = mean
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.init = init

    @staticmethod
    def _assign(points, centers):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def _initialize(self, points, rng):
        if self.init == "random":
            return points[rng.choice(len(points), self.n_clusters, replace=False)].copy()
        indices = [int(rng.integers(len(points)))]
        closest = ((points - points[indices[0]]) ** 2).sum(axis=1)
        for _ in range(1, self.n_clusters):
            total = closest.sum()
            if total > 0:
                index = int(rng.choice(len(points), p=closest / total))
            else:
                remaining = np.setdiff1d(np.arange(len(points)), indices)
                index = int(rng.choice(remaining))
            indices.append(index)
            closest = np.minimum(closest, ((points - points[index]) ** 2).sum(axis=1))
        return points[indices].copy()

    def fit(self, X):
        """Fit and return self; ``converged_`` reports the selected run's status."""
        X = check_array(X, dtype=float)
        if self.mean not in MEANS:
            raise ValueError(f"mean must be one of {MEANS}")
        if self.init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'")
        for name in ("n_clusters", "n_init", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_clusters > len(X):
            raise ValueError("n_clusters must not exceed the number of samples")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and non-negative")
        self.n_features_in_ = X.shape[1]
        minimum = X.min(axis=0)
        self.shift_ = (
            np.where(minimum <= 0, 1.0 - minimum, 0.0)
            if self.mean != "arithmetic" else np.zeros(X.shape[1])
        )
        points = X + self.shift_
        rng = np.random.default_rng(self.random_state)
        best = None
        self.n_iter_total_ = 0
        for _ in range(self.n_init):
            centers = self._initialize(points, rng)
            converged = False
            previous_labels = None
            for iteration in range(1, self.max_iter + 1):
                labels = self._assign(points, centers)
                updated = centers.copy()
                for cluster in range(self.n_clusters):
                    members = points[labels == cluster]
                    if len(members):
                        updated[cluster] = centroid(members, self.mean)
                empty = np.flatnonzero(np.bincount(labels, minlength=self.n_clusters) == 0)
                if len(empty):
                    occupied = np.setdiff1d(np.arange(self.n_clusters), empty)
                    distances = (
                        (points[:, None, :] - updated[occupied][None, :, :]) ** 2
                    ).sum(axis=2).min(axis=1)
                    for cluster in empty:
                        index = int(np.argmax(distances))
                        updated[cluster] = points[index]
                        distances = np.minimum(
                            distances, ((points - points[index]) ** 2).sum(axis=1),
                        )
                        distances[index] = -1
                movement = np.linalg.norm(updated - centers, axis=1).max()
                centers = updated
                if not len(empty) and (
                    movement <= self.tol or np.array_equal(labels, previous_labels)
                ):
                    converged = True
                    break
                previous_labels = labels
            labels = self._assign(points, centers)
            inertia = float(((points - centers[labels]) ** 2).sum())
            missing = self.n_clusters - len(np.unique(labels))
            self.n_iter_total_ += iteration
            if best is None or (missing, inertia) < (best[0], best[1]):
                best = (missing, inertia, centers.copy(), labels.copy(), iteration, converged)
        (
            self.empty_clusters_, self.inertia_, centers, self.labels_,
            self.n_iter_, self.converged_,
        ) = best
        self.cluster_centers_ = centers - self.shift_
        if self.empty_clusters_:
            warnings.warn(
                f"Selected KMeans run has {self.empty_clusters_} empty clusters",
                RuntimeWarning, stacklevel=2,
            )
        if not self.converged_:
            warnings.warn(
                "Selected KMeans run reached max_iter without convergence",
                RuntimeWarning, stacklevel=2,
            )
        return self

    def predict(self, X):
        """Assign new points in the original coordinate system."""
        check_is_fitted(self, "cluster_centers_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than fitted data")
        return self._assign(X, self.cluster_centers_)

    def fit_predict(self, X):
        return self.fit(X).labels_
