import numpy as np
import pytest

from pythagorean_kmeans import gap_statistic, within_cluster_dispersion
from pythagorean_kmeans.gap import fit_nonempty, select_k


def test_dispersion_matches_pairwise_definition():
    points = np.array([[1, 2], [3, 8], [20, 21], [23, 27]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    expected = 0.0
    for label in [0, 1]:
        members = points[labels == label]
        expected += ((members[:, None] - members[None, :]) ** 2).sum() / (2 * len(members))
    assert within_cluster_dispersion(points, labels) == pytest.approx(expected)


@pytest.mark.parametrize("mean", ["arithmetic", "geometric", "harmonic"])
def test_gap_is_deterministic_and_uses_one_standard_error_rule(mean):
    rng = np.random.default_rng(10)
    points = np.vstack([rng.normal(2, 0.1, (15, 2)), rng.normal(8, 0.1, (15, 2))])
    first = gap_statistic(points, range(1, 4), mean=mean, n_references=3, random_state=5)
    second = gap_statistic(points, range(1, 4), mean=mean, n_references=3, random_state=5)
    np.testing.assert_array_equal(first.gaps, second.gaps)
    np.testing.assert_array_equal(first.standard_errors, second.standard_errors)
    assert first.optimal_k == 2
    assert not first.selected_at_boundary
    index = first.optimal_k - 1
    assert first.gaps[index] >= first.gaps[index + 1] - first.standard_errors[index + 1]


def test_unresolved_gap_search_is_flagged():
    rng = np.random.default_rng(10)
    points = np.vstack([rng.normal(2, 0.1, (15, 2)), rng.normal(8, 0.1, (15, 2))])
    result = gap_statistic(points, range(1, 3), n_references=3, random_state=5)
    assert result.optimal_k == 2
    assert result.selected_at_boundary


def test_max_gap_selection():
    rng = np.random.default_rng(10)
    points = np.vstack([rng.normal(2, 0.1, (15, 2)), rng.normal(8, 0.1, (15, 2))])
    result = gap_statistic(
        points, range(1, 4), n_references=3, random_state=5, selection="max_gap",
    )
    assert result.optimal_k == result.k_values[np.argmax(result.gaps)]


def test_global_one_se_is_not_raw_maximum_or_local_one_se():
    values = np.arange(1, 5)
    gaps = np.array([0, 0.1, 1.0, 1.1])
    errors = np.array([0.05, 0.2, 0.05, 0.2])
    assert select_k(values, gaps, errors, "dudoit_fridlyand")[0] == 3
    assert select_k(values, gaps, errors, "max")[0] == 4
    assert select_k(values, gaps, errors, "one_se")[0] == 1


def test_uniform_data_selects_one_cluster():
    points = np.random.default_rng(42).uniform(size=(2000, 2))
    result = gap_statistic(points, range(1, 6), n_references=10, random_state=5)
    assert result.optimal_k == 1


def test_standard_error_uses_population_variance(monkeypatch):
    dispersions = iter([2, 1, np.e**2, 2, 1, np.e**2])
    monkeypatch.setattr(
        "pythagorean_kmeans.gap.within_cluster_dispersion",
        lambda data, labels: next(dispersions),
    )
    result = gap_statistic(np.arange(6).reshape(-1, 1), range(1, 3), n_references=2)
    np.testing.assert_allclose(result.standard_errors, np.sqrt(1.5))


def test_incomplete_fit_retries_with_new_seed_and_records_it(monkeypatch):
    from pythagorean_kmeans import gap

    original_fit = gap.KMeans.fit
    seeds = []

    def fit(self, data):
        model = original_fit(self, data)
        if self.n_clusters == 2:
            seeds.append(self.random_state)
            if len(seeds) == 1:
                model.labels_[:] = 0
        return model

    monkeypatch.setattr(gap.KMeans, "fit", fit)
    points = np.array([[0], [1], [2], [10], [11], [12]])
    result = gap_statistic(points, range(1, 3), n_references=2)
    assert len(result.empty_cluster_retries) == 1
    assert result.empty_cluster_retries[0]["k"] == 2
    assert result.empty_cluster_retries[0]["data_index"] == 0
    seeds.clear()
    model, retries = fit_nonempty(
        points, 2, mean="arithmetic", rng=np.random.default_rng(0),
    )
    assert len(np.unique(model.labels_)) == 2
    assert len(retries) == 1
    assert retries[0]["seed"] == seeds[0]
    assert retries[0]["occupied_clusters"] == 1
    assert retries[0]["retry_seed"] == seeds[1]
    assert seeds[0] != seeds[1]


def test_impossible_cluster_count_raises_after_bounded_retries():
    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            fit_nonempty(
                np.ones((5, 2)), 2, mean="arithmetic",
                rng=np.random.default_rng(0), max_retries=1, max_iter=2,
            )


@pytest.mark.parametrize(
    "kwargs", [{"k_values": [1, 3]}, {"n_references": 1}, {"selection": "other"}],
)
def test_invalid_gap_configuration(kwargs):
    with pytest.raises(ValueError):
        gap_statistic([[1], [2], [4], [8]], **kwargs)


def test_zero_dispersion_is_explicit_error():
    with pytest.raises(ValueError, match="positive dispersion"):
        gap_statistic(np.ones((5, 2)), range(1, 3), n_references=2)
