import csv
import importlib.util
import json
from pathlib import Path

import pytest


def load_compare():
    path = Path(__file__).resolve().parents[1] / "experiments" / "compare.py"
    spec = importlib.util.spec_from_file_location("compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iris_smoke_writes_dataset_scoped_outputs_without_network(tmp_path, monkeypatch):
    compare = load_compare()

    def forbid_network(*args, **kwargs):
        pytest.fail("Offline iris experiment attempted a network request")

    monkeypatch.setattr(compare, "urlopen", forbid_network)
    preserved = tmp_path / "gas_turbine" / "metadata.json"
    preserved.parent.mkdir()
    preserved.write_text("preserve me")
    compare.main([
        "--dataset", "iris", "--output-dir", str(tmp_path),
        "--gap-samples", "100", "--references", "2", "--max-k", "3",
    ])
    output = tmp_path / "iris"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["gap_selection"] == "dudoit_fridlyand"
    assert metadata["gap_n_init"] == 10
    assert preserved.read_text() == "preserve me"
    with (output / "selected_k.csv").open() as handle:
        selections = list(csv.DictReader(handle))
    assert len(selections) == 9
    for mean in compare.MEANS:
        assert metadata[mean]["occupied_clusters"] == metadata[mean]["optimal_k"]
        assert {row["rule"] for row in selections if row["mean"] == mean} == {
            "max", "one_se", "dudoit_fridlyand",
        }
    with (output / "gap.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 9
    for name in ("metrics.md", "comparison.svg", "arithmetic.svg", "geometric.svg", "harmonic.svg"):
        assert (output / name).stat().st_size > 0


def test_archive_hash_mismatch_is_explicit():
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_compare().validate_archive(b"not the pinned UCI archive")


def test_cached_archive_hash_is_checked(tmp_path, monkeypatch):
    compare = load_compare()
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    cache = tmp_path / ".cache" / "gas_turbine.zip"
    cache.parent.mkdir()
    cache.write_bytes(b"corrupt cached archive")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        compare.load_data("gas-turbine")
