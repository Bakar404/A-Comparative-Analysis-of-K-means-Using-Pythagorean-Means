import numpy as np
import pytest

from pythagorean_kmeans import KMeans, centroid


@pytest.mark.parametrize(
    "mean, expected",
    [
        ("arithmetic", [5.0, 10.0]),
        ("geometric", [3.0, 6.0]),
        ("harmonic", [1.8, 3.6]),
    ],
)
def test_centroid_and_single_cluster(mean, expected):
    points = np.array([[1.0, 2.0], [9.0, 18.0]])
    np.testing.assert_allclose(centroid(points, mean), expected)
    model = KMeans(1, mean=mean, random_state=7).fit(points)
    np.testing.assert_allclose(model.cluster_centers_[0], expected)


@pytest.mark.parametrize("init", ["k-means++", "random"])
@pytest.mark.parametrize("mean", ["arithmetic", "geometric", "harmonic"])
def test_convergence_and_seed(mean, init):
    points = np.array([[1, 1], [1, 2], [2, 1], [20, 20], [20, 21], [21, 20]])
    first = KMeans(2, mean=mean, random_state=42, init=init).fit(points)
    second = KMeans(2, mean=mean, random_state=42, init=init).fit(points)
    assert first.converged_
    assert first.n_iter_ < first.max_iter
    assert len(set(first.labels_)) == 2
    np.testing.assert_array_equal(first.labels_[:3], [first.labels_[0]] * 3)
    np.testing.assert_array_equal(first.labels_[3:], [first.labels_[3]] * 3)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)
    np.testing.assert_array_equal(first.predict(points), first.labels_)


@pytest.mark.parametrize(
    "mean, shifted_centers",
    [
        ("geometric", [[2, 2], [np.sqrt(110), np.sqrt(110)]]),
        ("harmonic", [[1.6, 1.6], [220 / 21, 220 / 21]]),
    ],
)
def test_non_positive_shift_and_prediction(mean, shifted_centers):
    points = np.array([[-2.0, 0.0], [1.0, 3.0], [7.0, 9.0], [8.0, 10.0]])
    model = KMeans(2, mean=mean, random_state=0).fit(points)
    np.testing.assert_array_equal(model.shift_, [3.0, 1.0])
    centers = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 0])]
    np.testing.assert_allclose(centers, np.array(shifted_centers) - [3, 1])
    predicted = model.predict([[-100.0, 0.0], [20.0, 22.0]])
    np.testing.assert_array_equal(predicted, [model.labels_[0], model.labels_[-1]])
    assert predicted[0] != predicted[1]
    with pytest.raises(ValueError, match="positive"):
        centroid(points, mean)


def test_empty_clusters_remain_finite():
    with pytest.warns(RuntimeWarning):
        model = KMeans(3, random_state=0, max_iter=3).fit(np.ones((4, 2)))
    assert model.empty_clusters_ == 2
    assert np.isfinite(model.cluster_centers_).all()
    assert model.inertia_ == 0


def test_empty_cluster_is_reseeded(monkeypatch):
    points = [[0], [0], [0], [10], [11]]
    monkeypatch.setattr(
        KMeans, "_initialize", lambda self, data, rng: data[[0, 1, 3]].copy(),
    )
    model = KMeans(3, init="random", random_state=0, n_init=1).fit(points)
    assert model.empty_clusters_ == 0
    assert len(np.unique(model.labels_)) == 3
    np.testing.assert_allclose(np.sort(model.cluster_centers_[:, 0]), [0, 10, 11])


def test_iteration_limit_is_reported():
    with pytest.warns(RuntimeWarning, match="max_iter"):
        model = KMeans(1, n_init=1, max_iter=1, tol=0, random_state=0).fit([[1], [9]])
    assert not model.converged_


@pytest.mark.parametrize(
    "kwargs", [{"mean": "other"}, {"n_clusters": 0}, {"n_init": 0}, {"init": "other"}],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        KMeans(**kwargs).fit([[1], [2], [3]])
