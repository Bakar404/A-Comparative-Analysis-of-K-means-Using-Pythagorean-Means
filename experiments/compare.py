"""Seeded reproduction; download the paper's public dataset on first use."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
from time import perf_counter
from urllib.request import urlopen
import zipfile

import numpy as np
import sklearn
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from pythagorean_kmeans import gap_statistic, within_cluster_dispersion
from pythagorean_kmeans.gap import fit_nonempty, select_k


ROOT = Path(__file__).resolve().parents[1]
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/551/"
    "gas+turbine+co+and+nox+emission+data+set.zip"
)
MEANS = ("arithmetic", "geometric", "harmonic")
DATA_SHA256 = "55fdb1acc25f05bd9c77aac285c424e9154ffb0dbf1bbadb56b69a1142231295"


def validate_archive(content):
    digest = hashlib.sha256(content).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"UCI archive SHA-256 mismatch: expected {DATA_SHA256}, got {digest}")
    with zipfile.ZipFile(io.BytesIO(content)) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("UCI archive failed integrity validation")
    return digest


def load_data(dataset):
    if dataset != "gas-turbine":
        data = {"iris": load_iris, "wine": load_wine}[dataset]()
        return data.data, {"source": f"scikit-learn built-in {dataset} (not in the paper)"}
    archive = ROOT / ".cache" / "gas_turbine.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        validate_archive(content)
        archive.write_bytes(content)
    content = archive.read_bytes()
    digest = validate_archive(content)
    with zipfile.ZipFile(io.BytesIO(content)) as zipped:
        arrays = [
            np.loadtxt(zipped.open(f"gt_{year}.csv"), delimiter=",", skiprows=1)
            for year in range(2011, 2016)
        ]
    data = np.vstack(arrays)
    if data.shape != (36733, 11) or not np.isfinite(data).all():
        raise ValueError(f"Unexpected UCI dataset shape or missing values: {data.shape}")
    return data, {
        "source": DATA_URL,
        "doi": "10.24432/C5WC95",
        "archive_sha256": digest,
    }


def write_scatter(path, points, models, shift, explained):
    """Write shared-axis PCA panels as SVG without a plotting dependency."""
    panel_width, height = 420, 380
    width = panel_width * len(models)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1e-12)
    lower = lower - 0.05 * span
    span = 1.1 * span
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#17202a">',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="17">'
        'Reproduction: PCA clustering (display subsample; crosses = centroids)</text>',
    ]
    for panel, (mean, model, labels) in enumerate(models):
        left = panel * panel_width + 60

        def project(values):
            normalized = (values - lower) / span
            return np.column_stack([
                left + 335 * normalized[:, 0], 305 - 235 * normalized[:, 1],
            ])

        svg.extend([
            f'<text x="{left + 167}" y="51" text-anchor="middle" font-size="15">'
            f'{mean.title()} | K = {model.n_clusters}</text>',
            f'<path d="M{left} 70 V305 H{left + 335}" fill="none" stroke="#52616b"/>',
            f'<text x="{left + 167}" y="351" text-anchor="middle" font-size="12">'
            f'PC1 ({explained[0]:.1%} variance)</text>',
            f'<text transform="translate({left - 43} 188) rotate(-90)" '
            f'text-anchor="middle" font-size="12">PC2 ({explained[1]:.1%} variance)</text>',
        ])
        for coordinate in range(2):
            for fraction in (0.0, 0.5, 1.0):
                value = lower[coordinate] + fraction * span[coordinate]
                if coordinate == 0:
                    x, y, anchor = left + 335 * fraction, 324, "middle"
                else:
                    x, y, anchor = left - 7, 309 - 235 * fraction, "end"
                svg.append(
                    f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                    f'font-size="10">{value:.1f}</text>'
                )
        for (x, y), label in zip(project(points), labels):
            color = f"hsl({int(label) * 137.508 % 360:.1f},65%,43%)"
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.7" fill="{color}" opacity="0.5"/>'
            )
        for label, (x, y) in enumerate(project(model.cluster_centers_ - shift)):
            svg.append(
                f'<path d="M{x - 5:.2f} {y:.2f} H{x + 5:.2f} M{x:.2f} {y - 5:.2f} '
                f'V{y + 5:.2f}" stroke="black" stroke-width="2"/>'
            )
            svg.append(f'<text x="{x + 6:.2f}" y="{y - 5:.2f}" font-size="10">{label}</text>')
    svg.extend(["</g>", "</svg>"])
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gas-turbine", "iris", "wine"), default="gas-turbine")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gap-samples", type=int, default=2000)
    parser.add_argument("--references", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument(
        "--selection", choices=("dudoit_fridlyand", "max", "one_se", "max_gap"),
        default="dudoit_fridlyand",
    )
    parser.add_argument("--n-init", type=int, default=10, help="Starts per gap-statistic fit")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets" / "results")
    args = parser.parse_args(argv)
    if args.gap_samples < 3 or args.references < 2 or args.max_k < 2:
        parser.error("gap-samples >= 3, references >= 2, and max-k >= 2 are required")
    if args.n_init < 1 or args.max_retries < 0:
        parser.error("n-init >= 1 and max-retries >= 0 are required")
    raw, provenance = load_data(args.dataset)
    pca = PCA(n_components=2, svd_solver="full")
    scores = pca.fit_transform(StandardScaler().fit_transform(raw))
    # The poster assumes positive inputs but does not specify a PCA-score shift.
    shift = np.maximum(1.0 - scores.min(axis=0), 0.0)
    positive = scores + shift
    sample_indices = np.random.default_rng(args.seed).choice(
        len(raw), min(args.gap_samples, len(raw)), replace=False,
    )
    sample = positive[sample_indices]
    if args.max_k >= len(sample):
        parser.error("max-k must be smaller than the gap sample count")
    output = args.output_dir / args.dataset.replace("-", "_")
    output.mkdir(parents=True, exist_ok=True)
    rows, models, gap_rows, selection_rows = [], [], [], []
    for mean in MEANS:
        print(f"Estimating gap statistic: {args.dataset}, {mean}", flush=True)
        gap = gap_statistic(
            sample, range(1, args.max_k + 1), mean=mean,
            n_references=args.references, random_state=args.seed,
            selection=args.selection, n_init=args.n_init, max_retries=args.max_retries,
        )
        start = perf_counter()
        model, fit_retries = fit_nonempty(
            positive, gap.optimal_k, mean=mean,
            rng=np.random.default_rng(args.seed), n_init=5, max_retries=args.max_retries,
        )
        seconds = perf_counter() - start
        labels = model.labels_
        sample_labels = labels[sample_indices]
        valid_full = 1 < len(np.unique(labels)) < len(labels)
        valid_sample = 1 < len(np.unique(sample_labels)) < len(sample_labels)
        silhouette = f"{silhouette_score(sample, sample_labels):.4f}" if valid_sample else "N/A"
        db = f"{davies_bouldin_score(positive, labels):.4f}" if valid_full else "N/A"
        ch = f"{calinski_harabasz_score(positive, labels):.2f}" if valid_full else "N/A"
        status = "yes" if model.converged_ else "no"
        rows.append(
            f"| {mean.title()} | {gap.optimal_k} | {silhouette} | {db} | {ch} | "
            f"{within_cluster_dispersion(positive, labels):.2f} | "
            f"{model.n_iter_} | {seconds / model.n_iter_total_:.4f} | {status} |"
        )
        models.append((mean, model, sample_labels))
        for k, value, error, dispersion in zip(
            gap.k_values, gap.gaps, gap.standard_errors, gap.dispersions,
        ):
            gap_rows.append([mean, int(k), float(value), float(error), float(dispersion)])
        for rule, selected in gap.selections.items():
            _, boundary = select_k(gap.k_values, gap.gaps, gap.standard_errors, rule)
            selection_rows.append([mean, rule, selected, boundary])
        provenance[mean] = {
            "optimal_k": gap.optimal_k,
            "selected_k_by_rule": gap.selections,
            "gap_empty_cluster_retries": gap.empty_cluster_retries,
            "fit_empty_cluster_retries": fit_retries,
            "selected_at_boundary": gap.selected_at_boundary,
            "occupied_clusters": int(len(np.unique(labels))),
            "inertia_to_variant_centers": model.inertia_,
            "fit_seconds": seconds,
            "n_iter": model.n_iter_,
            "n_iter_total": model.n_iter_total_,
            "converged": model.converged_,
        }
    table = "\n".join([
        "| Mean | K (gap) | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | "
        "Dispersion ↓ | Iterations | Seconds/iteration | Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        *rows,
    ])
    (output / "metrics.md").write_text(
        "# Reproduction metrics\n\n" + table + "\n\n"
        "Quality metrics use two standardized-feature PCA components. Gap uses a seeded "
        f"{len(sample)}-row subset; final fits and dispersion/DB/CH use all {len(raw)} rows. "
        "Silhouette and displayed points use the same subset. N/A means a partition has "
        "too few/many occupied clusters for that metric. Timing is wall time for all "
        "five starts divided by their total iterations, including initialization and "
        "final assignment overhead. Iterations reports the selected start only. "
        "See metadata.json for unresolved K-search boundaries.\n",
        encoding="utf-8",
    )
    with (output / "gap.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mean", "k", "gap", "standard_error", "dispersion"])
        writer.writerows(gap_rows)
    with (output / "selected_k.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mean", "rule", "selected_k", "selected_at_boundary"])
        writer.writerows(selection_rows)
    provenance.update({
        "label": "reproduction, not an exact replication",
        "dataset": args.dataset, "shape": list(raw.shape), "seed": args.seed,
        "gap_samples": len(sample), "n_references": args.references,
        "max_k": args.max_k, "gap_n_init": args.n_init, "fit_n_init": 5, "max_iter": 300,
        "init": "k-means++", "max_empty_cluster_retries": args.max_retries,
        "gap_standard_error_ddof": 0,
        "gap_selection": args.selection,
        "tol": 1e-4, "preprocessing": "StandardScaler, full-SVD PCA to 2 components",
        "positive_shift": shift.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    })
    (output / "metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    display = scores[sample_indices]
    write_scatter(output / "comparison.svg", display, models, shift, pca.explained_variance_ratio_)
    for mean, model, labels in models:
        write_scatter(
            output / f"{mean}.svg", display, [(mean, model, labels)],
            shift, pca.explained_variance_ratio_,
        )
    print(table)
    print(f"Saved reproduction artifacts to {output}")


if __name__ == "__main__":
    main()
