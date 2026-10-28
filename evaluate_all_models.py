from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from benchmark_data import (
    DATASET_SELECTIONS,
    expand_dataset_selection,
    load_benchmark_data,
)
from model.dca import ip_dca
from model.evaluate_mw_kmeans import (
    METRIC_FIELDS,
    RESULTS_DIR,
    resolve_cluster_count,
    score_axis_limits,
    score_clustering,
)
from model.gk_rdca import GKRDCA
from model.kmeans import kmeans
from model.mw_kmeans import MWKMeans


ALL_MODEL_FIELDS = (
    "model",
    *METRIC_FIELDS,
)


def evaluate_dataset(
    dataset_name: str,
    *,
    dataset_url: str | None = None,
    target_column: str | None = None,
    has_header: bool | None = None,
    include_target: bool = True,
    n_clusters: int | None = None,
    beta: float = 0.05,
    theta: float = 0.5,
    n_init: int = 10,
    max_iter: int = 300,
    dca_t: float = 10.0,
    dca_max_iter: int = 1000,
    gk_sigma: float = 1.0,
    gk_tau: float = 0.1,
    tol: float = 1e-6,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate every project clustering model on one loaded dataset."""
    load_options: dict[str, Any] = {
        "url": dataset_url,
        "target_column": target_column,
        "has_header": has_header,
    }
    if not include_target:
        load_options["include_target"] = False
    features, true_labels = load_benchmark_data(dataset_name, **load_options)
    n_clusters = resolve_cluster_count(true_labels, n_clusters)
    results: list[dict[str, Any]] = []

    mw_model = MWKMeans(
        n_clusters=n_clusters,
        beta=beta,
        theta=theta,
        max_iter=max_iter,
        remove_noise=False,
        n_init=n_init,
        random_state=random_state,
    )
    start = time.perf_counter()
    mw_labels = mw_model.fit_predict(features)
    mw_runtime = time.perf_counter() - start
    results.append(
        _score_model(
            "MW-K-means",
            dataset_name,
            true_labels,
            mw_labels,
            features,
            mw_runtime,
            mw_model.n_iter_,
        )
    )

    with threadpool_limits(limits=1):
        start = time.perf_counter()
        best_kmeans = None
        for init_index in range(n_init):
            candidate = kmeans(
                features,
                k=n_clusters,
                max_iter=max_iter,
                random_state=random_state + init_index,
            )
            if best_kmeans is None or candidate["inertia"] < best_kmeans["inertia"]:
                best_kmeans = candidate
        kmeans_runtime = time.perf_counter() - start

    if best_kmeans is None:
        raise RuntimeError("K-means did not produce a result")
    results.append(
        _score_model(
            "K-means",
            dataset_name,
            true_labels,
            best_kmeans["labels"],
            features,
            kmeans_runtime,
            best_kmeans["iterations"],
        )
    )

    start = time.perf_counter()
    dca_result = ip_dca(
        features,
        n_clusters=n_clusters,
        t=dca_t,
        max_iter=dca_max_iter,
        tol=tol,
        random_state=random_state,
        n_init=n_init,
    )
    dca_runtime = time.perf_counter() - start
    results.append(
        _score_model(
            "DCA",
            dataset_name,
            true_labels,
            dca_result["labels"],
            features,
            dca_runtime,
            dca_result["iterations"],
        )
    )

    gk_model = GKRDCA(
        n_clusters=n_clusters,
        sigma=gk_sigma,
        tau=gk_tau,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        random_state=random_state,
    )
    start = time.perf_counter()
    gk_labels = gk_model.fit_predict(features)
    gk_runtime = time.perf_counter() - start
    results.append(
        _score_model(
            "GK-RDCA",
            dataset_name,
            true_labels,
            gk_labels,
            features,
            gk_runtime,
            gk_model.n_iter_,
        )
    )
    return results


def _score_model(
    model_name: str,
    dataset_name: str,
    true_labels: np.ndarray | None,
    predicted_labels: np.ndarray,
    features: np.ndarray,
    runtime_seconds: float,
    iterations: int,
) -> dict[str, Any]:
    result = score_clustering(
        dataset_name,
        true_labels,
        predicted_labels,
        features=features,
        n_features=features.shape[1],
        n_clusters=len(np.unique(predicted_labels)),
        runtime_seconds=runtime_seconds,
        iterations=iterations,
    )
    return {"model": model_name, **result}


def save_combined_csv(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write all model and dataset results to one CSV file."""
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ALL_MODEL_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: result[field] for field in ALL_MODEL_FIELDS}
            for result in results
        )


def resolve_output_paths(
    name: str | None,
    output: Path | None,
    plot: Path | None,
) -> tuple[Path, Path]:
    base_name = Path(name or "all_models_results")
    if base_name.suffix.casefold() in {".csv", ".png"}:
        base_name = base_name.with_suffix("")
    if not base_name.name:
        raise ValueError("--name must contain a file name")

    csv_path = output if output is not None else RESULTS_DIR / f"{base_name}.csv"
    plot_path = plot if plot is not None else RESULTS_DIR / f"{base_name}.png"
    return csv_path, plot_path


def save_combined_plot(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save supervised metrics, Silhouette, and runtime comparisons as a PNG."""
    if not results:
        raise ValueError("At least one model result is required to create a plot")

    names = [f"{row['dataset']}\n{row['model']}" for row in results]
    positions = np.arange(len(results))

    figure, (metrics_axis, runtime_axis) = plt.subplots(
        2,
        1,
        figsize=(max(12, 1.8 * len(results)), 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    metric_names = ("Silhouette x100", "Accuracy", "Precision", "Recall", "F1")
    metric_fields = (
        "silhouette_score",
        "accuracy_percent",
        "precision_macro_percent",
        "recall_macro_percent",
        "f1_macro_percent",
    )
    metric_values = np.asarray(
        [
            [
                np.nan if row[field] is None else float(row[field])
                for field in metric_fields
            ]
            for row in results
        ]
    )
    metric_values[:, 0] *= 100
    figure.suptitle("Clustering model comparison", fontsize=15)
    bar_width = 0.1
    bar_spacing = 0.14
    for metric_index, metric_name in enumerate(metric_names):
        offset = (metric_index - (len(metric_names) - 1) / 2) * bar_spacing
        bars = metrics_axis.bar(
            positions + offset,
            metric_values[:, metric_index],
            bar_width,
            label=metric_name,
        )
        labels = [
            f"{height:.1f}" if np.isfinite(height) else ""
            for height in metric_values[:, metric_index]
        ]
        metrics_axis.bar_label(bars, labels=labels, fontsize=6, padding=2)

    metrics_axis.set_ylabel("Score (%) / normalized score x100")
    metrics_axis.set_ylim(*score_axis_limits(metric_values))
    metrics_axis.set_xticks(positions, names, rotation=0, ha="center")
    metrics_axis.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    metrics_axis.grid(axis="y", linestyle=":", alpha=0.5)

    runtimes = [row["runtime_seconds"] for row in results]
    runtime_bars = runtime_axis.bar(positions, runtimes, color="slateblue")
    runtime_axis.bar_label(runtime_bars, fmt="%.4f", fontsize=7, padding=2)
    runtime_axis.set_ylabel("Seconds")
    runtime_axis.set_title("Execution time")
    runtime_axis.set_xticks(positions, names, rotation=0, ha="center")
    runtime_axis.set_ylim(0, max(runtimes, default=0.0) * 1.2 or 1.0)
    runtime_axis.grid(axis="y", linestyle=":", alpha=0.5)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = Path(output_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MW-K-means, K-means, DCA, and GK-RDCA."
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        choices=DATASET_SELECTIONS,
        default="all",
        help="Built-in dataset to evaluate, or all datasets (default: all).",
    )
    dataset_group.add_argument(
        "--dataset-url",
        "--data-url",
        dest="dataset_url",
        help="Direct HTTP(S) URL to a tabular CSV dataset.",
    )
    parser.add_argument(
        "--target-column",
        help="Target column name or zero-based index (default: last column).",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Treat every CSV column as a feature and evaluate with Silhouette only.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Cluster count; required with --no-target, otherwise inferred from labels.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the CSV as headerless.",
    )
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--dca-t", type=float, default=10.0)
    parser.add_argument("--dca-max-iter", type=int, default=1000)
    parser.add_argument("--gk-sigma", type=float, default=1.0)
    parser.add_argument("--gk-tau", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--name",
        type=str,
        help=(
            "Name for a custom dataset in the results and base name for both "
            "CSV and PNG files."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit CSV output path; overrides the --name-derived path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Explicit comparison chart output path; overrides the --name-derived path.",
    )
    args = parser.parse_args()

    if args.n_init <= 0:
        parser.error("--n-init must be positive")
    if not args.dataset_url and (
        args.target_column is not None or args.no_header or args.no_target
    ):
        parser.error("--target-column, --no-header, and --no-target require --dataset-url")
    if args.no_target and args.target_column is not None:
        parser.error("--target-column cannot be used with --no-target")
    if args.no_target and args.n_clusters is None:
        parser.error("--n-clusters is required with --no-target")

    if args.dataset_url:
        dataset_names = (args.name or "custom",)
    else:
        dataset_names = expand_dataset_selection(args.dataset)
    results = [
        row
        for dataset_name in dataset_names
        for row in evaluate_dataset(
            dataset_name,
            dataset_url=args.dataset_url,
            target_column=args.target_column,
            has_header=False if args.no_header else None,
            include_target=not args.no_target,
            n_clusters=args.n_clusters,
            beta=args.beta,
            theta=args.theta,
            n_init=args.n_init,
            max_iter=args.max_iter,
            dca_t=args.dca_t,
            dca_max_iter=args.dca_max_iter,
            gk_sigma=args.gk_sigma,
            gk_tau=args.gk_tau,
            tol=args.tol,
            random_state=args.random_state,
        )
    ]
    try:
        output_path, plot_path = resolve_output_paths(
            args.name,
            args.output,
            args.plot,
        )
    except ValueError as error:
        parser.error(str(error))
    save_combined_csv(results, output_path)
    save_combined_plot(results, plot_path)
    print(f"Saved {len(results)} model results to: {output_path}")
    print(f"Saved comparison chart to: {plot_path}")
    print(
        "Dataset | Model | Silhouette | Accuracy % | Precision % | Recall % | "
        "F1 % | Time (s) | Iterations"
    )
    for result in results:
        metric_output = [
            "N/A" if result[field] is None else f"{result[field]:.2f}"
            for field in (
                "accuracy_percent",
                "precision_macro_percent",
                "recall_macro_percent",
                "f1_macro_percent",
            )
        ]
        silhouette = (
            "N/A"
            if result["silhouette_score"] is None
            else f"{result['silhouette_score']:.4f}"
        )
        print(
            f"{result['dataset']} | {result['model']} | "
            f"{silhouette} | "
            f"{' | '.join(metric_output)} | "
            f"{result['runtime_seconds']:.4f} | "
            f"{result['iterations']}"
        )


if __name__ == "__main__":
    main()
