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
from model.evaluate_mw_kmeans import (
    METRIC_FIELDS,
    RESULTS_DIR,
    print_results,
    resolve_cluster_count,
    save_csv,
    save_plot,
    score_clustering,
)
from model.gk_rdca import GKRDCA


GK_RDCA_EXTRA_FIELDS = (
    "objective_value",
    "converged",
    "empty_clusters",
    "sigma",
    "rho",
    "tau",
    "tol",
    "max_iter",
    "n_init",
    "random_state",
)
GK_RDCA_FIELDS = (*METRIC_FIELDS, *GK_RDCA_EXTRA_FIELDS)


def evaluate_dataset(
    dataset_name: str,
    *,
    dataset_url: str | None = None,
    target_column: str | None = None,
    has_header: bool | None = None,
    include_target: bool = True,
    n_clusters: int | None = None,
    sigma: float = 1.0,
    rho: float | None = None,
    tau: float = 0.1,
    max_iter: int = 300,
    tol: float = 1e-5,
    n_init: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit GK-RDCA and score the clustering, optionally without target labels."""
    load_options: dict[str, Any] = {
        "url": dataset_url,
        "target_column": target_column,
        "has_header": has_header,
    }
    if not include_target:
        load_options["include_target"] = False
    features, true_labels = load_benchmark_data(dataset_name, **load_options)
    n_clusters = resolve_cluster_count(true_labels, n_clusters)
    model = GKRDCA(
        n_clusters=n_clusters,
        sigma=sigma,
        rho=rho,
        tau=tau,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        random_state=random_state,
    )

    start = time.perf_counter()
    predicted_clusters = model.fit_predict(features)
    runtime_seconds = time.perf_counter() - start

    result = score_clustering(
        dataset_name,
        true_labels,
        predicted_clusters,
        features=features,
        n_features=features.shape[1],
        n_clusters=model.n_clusters,
        runtime_seconds=runtime_seconds,
        iterations=model.n_iter_,
    )
    result.update(
        {
            "objective_value": model.objective_value_,
            "converged": model.converged_,
            "empty_clusters": model.empty_clusters_,
            "sigma": model.sigma,
            "rho": model.rho_,
            "tau": model.tau,
            "tol": model.tol,
            "max_iter": model.max_iter,
            "n_init": model.n_init,
            "random_state": model.random_state,
        }
    )
    return result


def print_solver_details(results: list[dict[str, Any]]) -> None:
    """Print the GK-RDCA specific fields that print_results does not show."""
    print("\nGK-RDCA solver details")
    print("Dataset | F_GK | Converged | Empty clusters | sigma | rho | tau")
    for result in results:
        print(
            f"{result['dataset']} | {result['objective_value']:.6f} | "
            f"{result['converged']} | {result['empty_clusters']} | "
            f"{result['sigma']:g} | {result['rho']:g} | {result['tau']:g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GK-RDCA on Iris, Wine, Breast Cancer, and OpenML Yeast."
        )
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        choices=DATASET_SELECTIONS,
        default="all",
        help="Dataset to evaluate, or all available datasets (default: all).",
    )
    dataset_group.add_argument(
        "--dataset-url",
        "--data-url",
        dest="dataset_url",
        help="Direct HTTP(S) URL to a tabular CSV dataset.",
    )
    parser.add_argument("--target-column", help="Target column name or zero-based index.")
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Treat every CSV column as a feature and evaluate with Silhouette only.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the CSV as headerless.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Cluster count; required with --no-target, otherwise inferred from labels.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian kernel width; the paper uses values in [0.5, 10].",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="DC parameter rho >= 2 / sigma^2 (default: 2 / sigma^2).",
    )
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "gk_rdca_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=RESULTS_DIR / "gk_rdca_results.png",
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

    dataset_names = ("custom",) if args.dataset_url else expand_dataset_selection(args.dataset)
    results = [
        evaluate_dataset(
            name,
            dataset_url=args.dataset_url,
            target_column=args.target_column,
            has_header=False if args.no_header else None,
            include_target=not args.no_target,
            n_clusters=args.n_clusters,
            sigma=args.sigma,
            rho=args.rho,
            tau=args.tau,
            max_iter=args.max_iter,
            tol=args.tol,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        for name in dataset_names
    ]
    print_results(results)
    print_solver_details(results)
    save_csv(results, args.output, fieldnames=GK_RDCA_FIELDS)
    print(f"\nCSV exported to: {args.output}")
    save_plot(results, args.plot, title="GK-RDCA evaluation metrics")
    print(f"Chart exported to: {args.plot}")


if __name__ == "__main__":
    main()