from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from benchmark_data import (
    DATASET_SELECTIONS,
    expand_dataset_selection,
    load_benchmark_data,
)
from model.dca import ip_dca
from model.evaluate_mw_kmeans import (
    RESULTS_DIR,
    print_results,
    resolve_cluster_count,
    save_csv,
    save_plot,
    score_clustering,
)


def evaluate_dataset(
    dataset_name: str,
    *,
    dataset_url: str | None = None,
    target_column: str | None = None,
    has_header: bool | None = None,
    include_target: bool = True,
    n_clusters: int | None = None,
    t: float = 10.0,
    n_init: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run IP-DCA on one shared benchmark or URL-provided CSV dataset."""
    load_options: dict[str, Any] = {
        "url": dataset_url,
        "target_column": target_column,
        "has_header": has_header,
    }
    if not include_target:
        load_options["include_target"] = False
    features, true_labels = load_benchmark_data(dataset_name, **load_options)
    n_clusters = resolve_cluster_count(true_labels, n_clusters)

    start = time.perf_counter()
    result = ip_dca(
        features,
        n_clusters=n_clusters,
        t=t,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=n_init,
    )
    runtime_seconds = time.perf_counter() - start

    return score_clustering(
        dataset_name,
        true_labels,
        result["labels"],
        features=features,
        n_features=features.shape[1],
        n_clusters=n_clusters,
        runtime_seconds=runtime_seconds,
        iterations=result["iterations"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IP-DCA on shared datasets or a linked CSV dataset."
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        choices=DATASET_SELECTIONS,
        default="all",
        help="Built-in dataset to evaluate, or all available datasets (default: all).",
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
        help="Treat the CSV as headerless (useful for categorical data without headers).",
    )
    parser.add_argument("--t", type=float, default=10.0, help="DCA penalty parameter.")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "dca_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=RESULTS_DIR / "dca_results.png",
        help="Chart output path (PNG, SVG, or PDF).",
    )
    args = parser.parse_args()

    if not args.dataset_url and (
        args.target_column is not None or args.no_header or args.no_target
    ):
        parser.error("--target-column, --no-header, and --no-target require --dataset-url")
    if args.no_target and args.target_column is not None:
        parser.error("--target-column cannot be used with --no-target")
    if args.no_target and args.n_clusters is None:
        parser.error("--n-clusters is required with --no-target")

    if args.dataset_url:
        results = [
            evaluate_dataset(
                "custom",
                dataset_url=args.dataset_url,
                target_column=args.target_column,
                has_header=False if args.no_header else None,
                include_target=not args.no_target,
                n_clusters=args.n_clusters,
                t=args.t,
                n_init=args.n_init,
                max_iter=args.max_iter,
                tol=args.tol,
                random_state=args.random_state,
            )
        ]
    else:
        dataset_names = expand_dataset_selection(args.dataset)
        results = [
            evaluate_dataset(
                name,
                n_clusters=args.n_clusters,
                t=args.t,
                n_init=args.n_init,
                max_iter=args.max_iter,
                tol=args.tol,
                random_state=args.random_state,
            )
            for name in dataset_names
        ]

    print_results(results)
    save_csv(results, args.output)
    print(f"\nCSV exported to: {args.output}")
    save_plot(results, args.plot, title="IP-DCA evaluation metrics")
    print(f"Chart exported to: {args.plot}")


if __name__ == "__main__":
    main()
