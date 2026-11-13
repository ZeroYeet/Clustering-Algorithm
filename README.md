# Clustering Algorithm comparation

This project implements different clustering algorithms and compares them
with each other.

## Setup

```bash
pip install -r requirements.txt
```

## Evaluate datasets

```bash
python -m model.evaluate_mw_kmeans
python -m model.evaluate_classic_kmeans
python -m model.evaluate_dca
python -m model.evaluate_gk_rdca
python evaluate_all_models.py
```

The individual model scripts default
to all four datasets: Iris, Wine, Breast Cancer, and Yeast. Yeast is loaded
from the classic OpenML dataset ID 181 (1,484 samples, 8 numeric features,
10 classes); the first Yeast run requires internet access and scikit-learn
caches the download. `evaluate_all_models.py` runs all four models and writes
one row for each model and dataset to `results/all_models_results.csv` by
default. Select one dataset or all; the program also exports a comparison chart to
`results/all_models_results.png`. Silhouette is the primary clustering score
and does not require ground-truth labels. When labels are present, outputs also
include Accuracy, Precision, Recall, and F1. Those supervised metrics are
blank for unlabeled inputs. ARI and NMI are not reported. Comparison charts
scale Silhouette by 100 for comparison with percentage scores.
Silhouette is computed on the full data up to 2,000 rows and on a fixed,
reproducible sample of 2,000 rows for larger datasets.

```bash
python -m model.evaluate_mw_kmeans --dataset iris
python -m model.evaluate_classic_kmeans --dataset wine
python -m model.evaluate_mw_kmeans --dataset yeast
python -m model.evaluate_mw_kmeans --dataset all
python evaluate_all_models.py --dataset iris
```

To evaluate a tabular CSV hosted at a direct HTTP(S) download URL, provide
`--dataset-url` (or its alias `--data-url`) instead of `--dataset`. To evaluate
a local CSV, use `--dataset-file`; for example:

```bash
python evaluate_all_models.py --dataset-file data/my_data.csv --target-column "class"
python evaluate_all_models.py --dataset-file data/my_data.csv --no-target --n-clusters 3
```

The local CSV uses the same parsing and preprocessing rules as a URL-provided
CSV. The target column defaults to the last column unless `--no-target` is used.
The combined evaluator displays a progress bar as each of its four models
finishes running.

Target column can be selected by name or zero-based index with `--target-column`.
Numeric features are median-imputed, categorical features are mode-imputed and
one-hot encoded, and the resulting features are scaled to `[0, 1]`. Missing
target values are rejected. Columns whose names clearly identify them as IDs
(for example `User ID`) are excluded automatically. A header is required when
the feature columns are categorical; CSV, TSV, semicolon, and pipe delimiters
are detected automatically. Use `--no-header` for headerless files with
categorical features when automatic header detection is ambiguous. For example:

```bash
python -m model.evaluate_classic_kmeans --dataset-url "https://example.com/data.csv" --target-column "column_name"
python -m model.evaluate_mw_kmeans --dataset-url "https://example.com/data.csv" --target-column "column_name"
python -m model.evaluate_dca --dataset-url "https://example.com/data.csv" --target-column "column_name"
python -m model.evaluate_gk_rdca --dataset-url "https://example.com/data.csv" --target-column "column_name"
python evaluate_all_models.py --dataset-url "https://example.com/data.csv" --target-column "column_name"
```

This supports direct links to tabular files, not dataset landing pages (such as
a Kaggle dataset page). Numeric and categorical columns are supported; formats
such as Excel, archives, and images are not CSV tabular inputs.

For a linked CSV with no ground-truth label column, pass `--no-target` and
provide the desired number of clusters with `--n-clusters`. All columns are
then treated as features (identifier columns are still excluded), and only
Silhouette and operational metrics are
reported. For example:

```bash
python evaluate_all_models.py --dataset-url "https://example.com/unlabeled.csv" --no-target --n-clusters 3
python -m model.evaluate_gk_rdca --dataset-url "https://example.com/unlabeled.csv" --no-target --n-clusters 3
```

By default, linked CSV evaluation still treats the last column as the target
and infers the cluster count from its distinct values. `--target-column` can
select a different label column.


The combined CSV and PNG file names can be set together with `--name`; both
are saved under `results/` by default. For example,
`python evaluate_all_models.py --name bank_results` creates
`results/bank_results.csv` and `results/bank_results.png`. When evaluating a
custom CSV URL, `--name` also replaces the `custom` dataset label shown in the
CSV and chart; without it, that label remains `custom`. For example:

```bash
python evaluate_all_models.py --data-url "https://example.com/bank.csv" --no-target --n-clusters 3 --name bank
```

If the name ends in `.csv` or `.png`, that extension is removed before
generating both files. Use `--output` or `--plot` to override either output
path individually. Missing output directories are created automatically.


## Notes

- `beta` controls the density threshold: `MinPts = ceil(beta * N)`
- `theta` scales neighborhood radius `epsilon` based on average pairwise distance
- `MWKMeans` treats low-density points as noise by default; the benchmark sets
  `remove_noise=False` so every point is included in the evaluation
- MW-K-means computes the mean pairwise distance in triangular chunks and
  updates nearest-center distances incrementally; density estimation remains
  quadratic in the number of samples
- Multiple initializations choose using clustering inertia only, without
  consulting dataset labels; better inertia does not guarantee higher
  Silhouette or accuracy on every dataset
- IP-DCA uses k-means++ center initialization and can run multiple restarts;
  it retains the run with the lowest DCA penalized objective, without consulting
  ground-truth labels. Set the restart count with `--n-init`.
- Classic K-means uses the project's implementation in `kmeans.py` and runs
  10 random initializations by default, selecting the result with lowest inertia

Yeast source: [OpenML dataset 181](https://www.openml.org/d/181).
