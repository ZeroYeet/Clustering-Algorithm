from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark_data
import model.evaluate_classic_kmeans as evaluate_classic_kmeans
import model.evaluate_mw_kmeans as evaluate_mw_kmeans
import evaluate_all_models as evaluate_all_models
import model.mw_kmeans as mw_kmeans
import model.evaluate_dca as evaluate_dca
import model.evaluate_gk_rdca as evaluate_gk_rdca
from benchmark_data import expand_dataset_selection
from model.evaluate_classic_kmeans import evaluate_dataset as evaluate_classic_dataset
from model.evaluate_dca import evaluate_dataset as evaluate_dca_dataset
from model.dca import ip_dca
from model.evaluate_mw_kmeans import (
    RESULTS_DIR,
    evaluate_dataset,
    resolve_cluster_count,
    save_csv,
    save_plot,
    score_axis_limits,
    score_clustering,
)
from model.mw_kmeans import MWKMeans


def test_mw_kmeans_clusters_two_blobs():
    rng = np.random.default_rng(7)
    cluster_a = rng.normal(loc=[0.0, 0.0], scale=0.4, size=(60, 2))
    cluster_b = rng.normal(loc=[5.0, 5.0], scale=0.4, size=(60, 2))
    X = np.vstack([cluster_a, cluster_b])

    model = MWKMeans(n_clusters=2, beta=0.05, theta=0.6, max_iter=200, tol=1e-6)
    labels = model.fit_predict(X)

    unique_labels = set(labels.tolist())
    assert unique_labels.issubset({-1, 0, 1})
    assert len(set(labels[labels != -1])) == 2
    assert model.cluster_centers_.shape == (2, 2)
    assert model.weights_.shape == (2,)


def test_pairwise_distances_matches_euclidean_distance():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]])
    distances = MWKMeans(n_clusters=2)._pairwise_distances(X)

    np.testing.assert_allclose(
        distances,
        np.array([[0.0, 5.0, np.sqrt(5)], [5.0, 0.0, np.sqrt(20)], [np.sqrt(5), np.sqrt(20), 0.0]]),
    )


def test_mean_pairwise_distance_matches_upper_triangle_mean_across_chunks(monkeypatch):
    monkeypatch.setattr(mw_kmeans, "_PAIRWISE_CHUNK_SIZE", 2)
    X = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [-1.0, 2.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )
    model = MWKMeans(n_clusters=2)
    expected = model._pairwise_distances(X)[np.triu_indices(len(X), k=1)].mean()

    assert model._mean_pairwise_distance(X) == pytest.approx(expected)


def test_initial_centers_keep_greedy_max_min_selection_with_duplicate_rows():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [5.0, 0.0],
            [5.0, 0.0],
            [0.0, 2.0],
        ]
    )
    density = np.array([10, 0, 0, 0, 0])
    centers = MWKMeans(n_clusters=3)._select_initial_centers(X, density)

    np.testing.assert_array_equal(
        centers,
        np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 2.0]]),
    )


def test_weighted_distance_matrix_matches_per_center_distances():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]])
    centers = np.array([[1.0, 1.0], [2.0, 3.0]])
    weights = np.array([0.25, 0.75])
    model = MWKMeans(n_clusters=2)
    expected = np.column_stack(
        [
            model._weighted_distance(X, center, weights) ** 2
            for center in centers
        ]
    )

    np.testing.assert_allclose(
        model._weighted_distance_matrix(X, centers, weights),
        expected,
    )


def test_fit_computes_density_parameters_once(monkeypatch):
    model = MWKMeans(n_clusters=2, remove_noise=False)
    original_infer = model._infer_density_params
    calls = 0

    def counted_infer(X):
        nonlocal calls
        calls += 1
        return original_infer(X)

    monkeypatch.setattr(model, "_infer_density_params", counted_infer)
    model.fit(np.array([[0.0], [0.1], [5.0], [5.1]]))

    assert calls == 1


def test_multiple_initializations_are_reproducible_and_not_worse_than_first():
    rng = np.random.default_rng(21)
    X = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.8, size=(45, 2)),
            rng.normal(loc=[3.0, 0.0], scale=0.8, size=(45, 2)),
            rng.normal(loc=[1.5, 2.5], scale=0.8, size=(45, 2)),
        ]
    )
    single_start = MWKMeans(n_clusters=3, remove_noise=False, n_init=1)
    multi_start = MWKMeans(
        n_clusters=3,
        remove_noise=False,
        n_init=8,
        random_state=12,
    )

    single_start.fit(X)
    multi_start.fit(X)
    repeated_fit = MWKMeans(
        n_clusters=3,
        remove_noise=False,
        n_init=8,
        random_state=12,
    ).fit(X)

    assert multi_start.inertia_ <= single_start.inertia_ + 1e-12
    assert multi_start.inertia_ == pytest.approx(repeated_fit.inertia_)
    np.testing.assert_array_equal(multi_start.labels_, repeated_fit.labels_)
    np.testing.assert_array_equal(multi_start.labels_, multi_start.predict(X))


def test_reported_inertia_matches_final_assignments():
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.7, size=(40, 2)),
            rng.normal(loc=[3.0, 0.0], scale=0.7, size=(40, 2)),
            rng.normal(loc=[1.5, 2.5], scale=0.7, size=(40, 2)),
        ]
    )
    model = MWKMeans(n_clusters=3, remove_noise=False, n_init=12).fit(X)
    weighted_data = X * np.sqrt(model.weights_)
    weighted_centers = model.cluster_centers_ * np.sqrt(model.weights_)
    distances = model._squared_distance_matrix(
        weighted_data,
        weighted_centers,
    )
    selected_distances = distances[np.arange(len(X)), model.labels_]

    assert model.inertia_ == pytest.approx(selected_distances.sum())


def test_mw_kmeans_rejects_non_positive_n_init():
    with pytest.raises(ValueError, match="n_init must be positive"):
        MWKMeans(n_clusters=2, n_init=0)


def test_dca_multiple_initializations_are_reproducible_and_select_best_objective():
    rng = np.random.default_rng(12)
    X = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.25, size=(30, 2)),
            rng.normal(loc=[4.0, 4.0], scale=0.25, size=(30, 2)),
            rng.normal(loc=[0.0, 4.0], scale=0.25, size=(30, 2)),
        ]
    )
    single_result = ip_dca(X, n_clusters=3, max_iter=100, n_init=1)
    multi_result = ip_dca(
        X,
        n_clusters=3,
        max_iter=100,
        random_state=21,
        n_init=5,
    )
    repeated_result = ip_dca(
        X,
        n_clusters=3,
        max_iter=100,
        random_state=21,
        n_init=5,
    )

    assert multi_result["penalized_objective"] <= single_result["penalized_objective"]
    assert multi_result["penalized_objective"] == pytest.approx(
        repeated_result["penalized_objective"]
    )
    np.testing.assert_array_equal(multi_result["labels"], repeated_result["labels"])


def test_dca_rejects_non_positive_n_init():
    with pytest.raises(ValueError, match="n_init must be positive"):
        ip_dca(np.array([[0.0], [1.0]]), n_clusters=2, n_init=0)


def test_mw_kmeans_predict_uses_trained_centers():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.0, 5.1],
    ])

    model = MWKMeans(n_clusters=2, beta=0.35, theta=0.8)
    model.fit(X)
    preds = model.predict(np.array([[0.05, 0.05], [4.9, 5.1]]))

    assert preds.shape == (2,)
    assert set(preds.tolist()).issubset({0, 1})


def test_evaluate_iris_and_wine_returns_complete_dataset_metrics():
    expected_data = {
        "iris": (150, 3, 4),
        "wine": (178, 3, 13),
        "cancer": (569, 2, 30),
    }

    for dataset_name, (sample_count, cluster_count, feature_count) in expected_data.items():
        result = evaluate_dataset(dataset_name)

        assert result["samples"] == sample_count
        assert result["clusters"] == cluster_count
        assert result["features"] == feature_count
        assert result["noise_points"] == 0
        for metric in (
            "silhouette_score",
            "accuracy_percent",
            "precision_macro_percent",
            "recall_macro_percent",
            "f1_macro_percent",
        ):
            assert 0 <= result[metric] <= 100
        assert result["runtime_seconds"] >= 0
        assert result["iterations"] > 0


def test_both_iris_evaluators_use_all_features():
    mw_result = evaluate_dataset("iris")
    classic_result = evaluate_classic_dataset("iris")

    assert mw_result["features"] == classic_result["features"] == 4


def test_gk_rdca_evaluator_reports_silhouette_score():
    result = evaluate_gk_rdca.evaluate_dataset(
        "iris",
        n_init=1,
        max_iter=10,
    )

    assert result["samples"] == 150
    assert result["features"] == 4
    assert -1 <= result["silhouette_score"] <= 1
    assert result["objective_value"] >= 0


def test_evaluate_dataset_rejects_unknown_dataset():
    try:
        evaluate_dataset("unknown")
    except ValueError as error:
        assert "iris, wine, cancer" in str(error)
    else:
        raise AssertionError("Expected ValueError for an unknown dataset")


def test_dataset_selection_includes_supported_datasets_only():
    assert expand_dataset_selection("all") == ("iris", "wine", "cancer", "yeast")
    assert expand_dataset_selection("iris") == ("iris",)
    for removed_dataset in ("s1", "a1", "s-sets", "a-sets", "uef"):
        with pytest.raises(ValueError):
            expand_dataset_selection(removed_dataset)


def test_load_openml_yeast_dataset_by_fixed_id(monkeypatch):
    raw_features = np.arange(1484 * 8, dtype=float).reshape(1484, 8)
    raw_features %= 101
    labels = np.asarray([f"class-{index % 10}" for index in range(1484)])
    calls = []

    def fake_fetch_openml(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=raw_features, target=labels)

    monkeypatch.setattr(benchmark_data, "fetch_openml", fake_fetch_openml)
    features, loaded_labels = benchmark_data.load_benchmark_data("yeast")

    assert calls == [{"data_id": 181, "as_frame": False, "parser": "auto"}]
    assert features.shape == (1484, 8)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features.min(axis=0), 0)
    np.testing.assert_allclose(features.max(axis=0), 1)
    np.testing.assert_array_equal(loaded_labels, labels)


def test_load_csv_dataset_from_url(monkeypatch):
    response = BytesIO(b"feature_a,feature_b,class\n0,10,alpha\n5,20,beta\n10,30,alpha\n")
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return response

    monkeypatch.setattr(benchmark_data, "urlopen", fake_urlopen)
    features, labels = benchmark_data.load_benchmark_data(
        url="https://example.com/data.csv"
    )

    assert calls == [("https://example.com/data.csv", 30)]
    np.testing.assert_allclose(features, [[0, 0], [0.5, 0.5], [1, 1]])
    np.testing.assert_array_equal(labels, ["alpha", "beta", "alpha"])


def test_github_blob_csv_url_is_converted_to_raw_download(monkeypatch):
    response = BytesIO(b"feature_a,feature_b,class\n0,10,alpha\n5,20,beta\n")
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return response

    monkeypatch.setattr(benchmark_data, "urlopen", fake_urlopen)
    features, labels = benchmark_data.load_benchmark_data(
        url=(
            "https://github.com/dphi-official/Datasets/blob/master/"
            "titanic_data.csv"
        )
    )

    assert calls == [
        (
            "https://raw.githubusercontent.com/dphi-official/Datasets/"
            "master/titanic_data.csv",
            30,
        )
    ]
    assert features.shape == (2, 2)
    np.testing.assert_array_equal(labels, ["alpha", "beta"])


def test_load_csv_without_target_uses_all_feature_columns(monkeypatch):
    response = BytesIO(b"feature_a,feature_b\n0,10\n5,20\n10,30\n")
    monkeypatch.setattr(benchmark_data, "urlopen", lambda url, timeout: response)

    features, labels = benchmark_data.load_benchmark_data(
        url="https://example.com/unlabeled.csv",
        include_target=False,
    )

    assert features.shape == (3, 2)
    assert labels is None
    np.testing.assert_allclose(features, [[0, 0], [0.5, 0.5], [1, 1]])


def test_silhouette_is_available_without_ground_truth_labels():
    features = np.array([[0.0], [0.1], [0.9], [1.0]])
    result = score_clustering(
        "unlabeled",
        None,
        np.array([0, 0, 1, 1]),
        features=features,
        n_features=1,
        n_clusters=2,
        runtime_seconds=0.1,
        iterations=3,
    )

    assert result["silhouette_score"] > 0.85
    assert result["accuracy_percent"] is None
    assert resolve_cluster_count(None, 2) == 2
    with pytest.raises(ValueError, match="n_clusters must be provided"):
        resolve_cluster_count(None, None)


def test_score_clustering_keeps_degenerate_cluster_result_with_no_silhouette():
    features = np.array([[0.0], [0.2], [0.7], [1.0]])
    result = score_clustering(
        "unlabeled",
        None,
        np.array([0, 0, 0, 0]),
        features=features,
        n_features=1,
        n_clusters=3,
        runtime_seconds=0.1,
        iterations=5,
    )

    assert result["clusters"] == 1
    assert result["silhouette_score"] is None


def test_large_dataset_silhouette_uses_reproducible_sample(monkeypatch):
    features = np.zeros((2001, 2))
    labels = np.tile([0, 1], 1001)[:2001]
    calls = []

    def fake_silhouette_score(X, predicted, *, sample_size, random_state):
        calls.append((X.shape, predicted.shape, sample_size, random_state))
        return 0.25

    monkeypatch.setattr(
        evaluate_mw_kmeans,
        "silhouette_score",
        fake_silhouette_score,
    )
    result = score_clustering(
        "large",
        None,
        labels,
        features=features,
        n_features=2,
        n_clusters=2,
        runtime_seconds=0.1,
        iterations=5,
    )

    assert calls == [((2001, 2), (2001,), 2000, 42)]
    assert result["silhouette_score"] == 0.25


def test_score_chart_limits_start_at_zero_for_non_negative_scores():
    assert score_axis_limits(np.array([[48.0, 72.0], [91.0, 63.0]])) == (
        0.0,
        105.0,
    )


def test_score_chart_limits_include_negative_silhouette_scores():
    lower, upper = score_axis_limits(np.array([[-20.0, 70.0], [40.0, 95.0]]))

    assert lower < -20.0
    assert upper == 105.0


def test_load_csv_preprocesses_categorical_missing_and_identifier_columns(monkeypatch):
    response = BytesIO(
        b"User ID,Gender,Age,EstimatedSalary,Purchased\n"
        b"15624510,Male,19,19000,0\n"
        b"15810944,Female,35,?,1\n"
        b"15668575,Male,26,43000,0\n"
        b"15603246,Female,27,57000,1\n"
    )
    def fake_urlopen(url, timeout):
        assert url == "https://example.com/Social_Network_Ads.csv"
        assert timeout == 30
        return response

    monkeypatch.setattr(benchmark_data, "urlopen", fake_urlopen)

    features, labels = benchmark_data.load_benchmark_data(
        url="https://example.com/Social_Network_Ads.csv",
        target_column="Purchased",
    )

    assert features.shape == (4, 4)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features.min(axis=0), 0)
    np.testing.assert_allclose(features.max(axis=0), 1)
    np.testing.assert_array_equal(labels, ["0", "1", "0", "1"])


def test_load_csv_supports_semicolon_delimiter_and_target_index(monkeypatch):
    response = BytesIO(b"red;1;A\nblue;3;B\nred;2;A\n")

    def fake_urlopen(url, timeout):
        assert url == "https://example.com/data.csv"
        assert timeout == 30
        return response

    monkeypatch.setattr(benchmark_data, "urlopen", fake_urlopen)

    features, labels = benchmark_data.load_benchmark_data(
        url="https://example.com/data.csv",
        target_column="2",
        has_header=False,
    )

    assert features.shape == (3, 3)
    np.testing.assert_array_equal(labels, ["A", "B", "A"])


def test_load_csv_dataset_from_url_rejects_non_http_urls():
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        benchmark_data.load_benchmark_data(url="file:///local/data.csv")


def test_dca_evaluates_dataset_with_ground_truth_cluster_count(monkeypatch):
    features = np.array([[0.0], [0.1], [0.9], [1.0]])
    labels = np.array(["left", "left", "right", "right"])
    calls = []

    def fake_load_benchmark_data(
        dataset_name, url=None, target_column=None, has_header=None
    ):
        assert dataset_name == "custom"
        assert url == "https://example.com/data.csv"
        assert target_column is None
        assert has_header is None
        return features, labels

    monkeypatch.setattr(
        evaluate_dca, "load_benchmark_data", fake_load_benchmark_data
    )

    def fake_ip_dca(X, **kwargs):
        calls.append((X, kwargs))
        return {"labels": np.array([0, 0, 1, 1]), "iterations": 7}

    monkeypatch.setattr(evaluate_dca, "ip_dca", fake_ip_dca)
    result = evaluate_dca_dataset("custom", dataset_url="https://example.com/data.csv")

    assert calls[0][1] == {
        "n_clusters": 2,
        "t": 10.0,
        "max_iter": 1000,
        "tol": 1e-6,
        "random_state": 42,
        "n_init": 10,
    }
    assert result["dataset"] == "custom"
    assert result["samples"] == 4
    assert result["iterations"] == 7
    assert result["accuracy_percent"] == 100.0


def test_both_evaluators_run_openml_yeast_without_network(monkeypatch):
    rng = np.random.default_rng(17)
    features = rng.random((1484, 8))
    labels = np.asarray([f"class-{index % 10}" for index in range(1484)])

    def fake_fetch_openml(data_id, as_frame, parser):
        assert (data_id, as_frame, parser) == (181, False, "auto")
        return SimpleNamespace(data=features, target=labels)

    monkeypatch.setattr(
        benchmark_data,
        "fetch_openml",
        fake_fetch_openml,
    )

    mw_result = evaluate_dataset("yeast")
    classic_result = evaluate_classic_dataset("yeast", n_init=2)

    for result in (mw_result, classic_result):
        assert result["samples"] == 1484
        assert result["features"] == 8
        assert result["clusters"] == 10
        assert result["noise_points"] == 0


def test_classic_kmeans_evaluates_all_shared_datasets():
    expected_data = {"iris": (150, 3), "wine": (178, 3), "cancer": (569, 2)}

    for dataset_name, (sample_count, cluster_count) in expected_data.items():
        result = evaluate_classic_dataset(dataset_name)

        assert result["samples"] == sample_count
        assert result["clusters"] == cluster_count
        assert result["noise_points"] == 0
        assert 0 <= result["accuracy_percent"] <= 100
        assert result["runtime_seconds"] >= 0
        assert result["iterations"] > 0


def test_classic_kmeans_rejects_invalid_n_init():
    try:
        evaluate_classic_dataset("iris", n_init=0)
    except ValueError as error:
        assert "n_init must be positive" in str(error)
    else:
        raise AssertionError("Expected ValueError for n_init=0")


def test_classic_evaluator_uses_project_kmeans_and_selects_best_restart(monkeypatch):
    features = np.array([[0.0], [0.1], [0.9], [1.0]])
    labels = np.array(["left", "left", "right", "right"])
    calls = []
    monkeypatch.setattr(
        evaluate_classic_kmeans,
        "load_benchmark_data",
        lambda dataset_name, **kwargs: (features, labels),
    )

    def fake_kmeans(X, k, max_iter, random_state):
        calls.append((X, k, max_iter, random_state))
        if random_state == 11:
            return {
                "labels": np.array([1, 1, 0, 0]),
                "inertia": 0.1,
                "iterations": 3,
            }
        return {
            "labels": np.array([0, 1, 0, 1]),
            "inertia": 2.0,
            "iterations": 2,
        }

    monkeypatch.setattr(evaluate_classic_kmeans, "kmeans", fake_kmeans)
    result = evaluate_classic_dataset(
        "custom",
        n_init=2,
        random_state=10,
        max_iter=50,
    )

    assert [call[1:] for call in calls] == [(2, 50, 10), (2, 50, 11)]
    assert result["iterations"] == 3
    assert result["accuracy_percent"] == 100.0


def test_combined_evaluator_runs_all_models_on_one_loaded_dataset(monkeypatch):
    features = np.array([[0.0], [0.1], [0.9], [1.0]])
    labels = np.array(["left", "left", "right", "right"])
    load_calls = []
    model_calls = []

    def fake_load(dataset_name, **kwargs):
        load_calls.append((dataset_name, kwargs))
        if kwargs.get("include_target") is False:
            return features, None
        return features, labels

    class FakeMWKMeans:
        def __init__(self, **kwargs):
            model_calls.append(("MW-K-means", kwargs))
            self.n_iter_ = 4

        def fit_predict(self, X):
            return np.array([0, 0, 1, 1])

    def fake_kmeans(X, **kwargs):
        model_calls.append(("K-means", kwargs))
        return {
            "labels": np.array([1, 1, 0, 0]),
            "inertia": 0.1,
            "iterations": 3,
        }

    def fake_ip_dca(X, **kwargs):
        model_calls.append(("DCA", kwargs))
        return {"labels": np.array([0, 0, 1, 1]), "iterations": 5}

    class FakeGKRDCA:
        def __init__(self, **kwargs):
            model_calls.append(("GK-RDCA", kwargs))
            self.n_iter_ = 6

        def fit_predict(self, X):
            return np.array([0, 0, 1, 1])

    monkeypatch.setattr(evaluate_all_models, "load_benchmark_data", fake_load)
    monkeypatch.setattr(evaluate_all_models, "MWKMeans", FakeMWKMeans)
    monkeypatch.setattr(evaluate_all_models, "kmeans", fake_kmeans)
    monkeypatch.setattr(evaluate_all_models, "ip_dca", fake_ip_dca)
    monkeypatch.setattr(evaluate_all_models, "GKRDCA", FakeGKRDCA)

    results = evaluate_all_models.evaluate_dataset(
        "custom",
        dataset_url="https://example.com/data.csv",
        target_column="class",
        n_init=1,
    )

    assert len(load_calls) == 1
    assert [result["model"] for result in results] == [
        "MW-K-means",
        "K-means",
        "DCA",
        "GK-RDCA",
    ]
    assert all(result["samples"] == 4 for result in results)
    assert [result["iterations"] for result in results] == [4, 3, 5, 6]
    dca_call = next(call for call in model_calls if call[0] == "DCA")
    assert dca_call[1]["n_init"] == 1

    unlabeled_results = evaluate_all_models.evaluate_dataset(
        "custom",
        dataset_url="https://example.com/data.csv",
        include_target=False,
        n_clusters=2,
        n_init=1,
    )
    assert len(unlabeled_results) == 4
    assert all(row["silhouette_score"] > 0.85 for row in unlabeled_results)
    assert all(row["accuracy_percent"] is None for row in unlabeled_results)


def test_combined_csv_contains_all_models_in_one_file(tmp_path):
    rows = [
        {
            "model": model,
            **{
                field: (
                    "iris"
                    if field == "dataset"
                    else 4
                    if field in {"features", "iterations"}
                    else 0
                )
                for field in evaluate_all_models.METRIC_FIELDS
            },
        }
        for model in ("MW-K-means", "K-means", "DCA", "GK-RDCA")
    ]
    output_path = tmp_path / "results" / "all_models.csv"

    evaluate_all_models.save_combined_csv(rows, output_path)

    exported = output_path.read_text(encoding="utf-8").splitlines()
    assert exported[0].startswith("model,dataset,samples,features,")
    assert ",ari," not in exported[0]
    assert ",nmi," not in exported[0]
    assert [row.split(",")[0] for row in exported[1:]] == [
        "MW-K-means",
        "K-means",
        "DCA",
        "GK-RDCA",
    ]


@pytest.mark.parametrize(
    "name",
    ["bank_results", "bank_results.csv", "bank_results.png"],
)
def test_name_resolves_csv_and_png_paths_from_same_base(name):
    csv_path, plot_path = evaluate_all_models.resolve_output_paths(
        name,
        output=None,
        plot=None,
    )

    assert csv_path == RESULTS_DIR / "bank_results.csv"
    assert plot_path == RESULTS_DIR / "bank_results.png"


def test_explicit_combined_output_paths_override_name_paths(tmp_path):
    csv_override = tmp_path / "custom.csv"
    png_override = tmp_path / "custom.png"

    csv_path, plot_path = evaluate_all_models.resolve_output_paths(
        "bank_results",
        output=csv_override,
        plot=png_override,
    )

    assert csv_path == csv_override
    assert plot_path == png_override


def test_combined_plot_exports_png_without_ari_or_nmi(tmp_path):
    rows = [
        {
            "model": model,
            "dataset": "iris",
            "accuracy_percent": 80.0,
            "silhouette_score": 0.75,
            "precision_macro_percent": 79.0,
            "recall_macro_percent": 78.0,
            "f1_macro_percent": 77.0,
            "runtime_seconds": 0.01,
        }
        for model in ("MW-K-means", "K-means", "DCA", "GK-RDCA")
    ]
    output_path = tmp_path / "charts" / "combined.png"

    evaluate_all_models.save_combined_plot(rows, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_combined_plot_handles_unlabeled_metrics(tmp_path):
    row = {
        "model": "GK-RDCA",
        "dataset": "custom",
        "silhouette_score": 0.6,
        "accuracy_percent": None,
        "precision_macro_percent": None,
        "recall_macro_percent": None,
        "f1_macro_percent": None,
        "runtime_seconds": 0.01,
    }
    output_path = tmp_path / "charts" / "unlabeled.png"

    evaluate_all_models.save_combined_plot([row], output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_csv_exports_evaluation_metrics(tmp_path):
    result = evaluate_dataset("iris")
    output_path = tmp_path / "results" / "evaluation.csv"

    save_csv([result], output_path)

    exported = output_path.read_text(encoding="utf-8").splitlines()
    assert exported[0].startswith("dataset,samples,features,clusters,")
    assert exported[1].startswith("iris,150,4,3,")


def test_save_plot_exports_chart_image(tmp_path):
    results = [evaluate_dataset("iris"), evaluate_dataset("cancer")]
    output_path = tmp_path / "results" / "metrics.png"

    save_plot(results, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_default_output_paths_point_to_results_directory():
    assert RESULTS_DIR.name == "results"
    assert RESULTS_DIR.parent == Path(__file__).resolve().parents[1]


def test_save_plot_rejects_empty_results(tmp_path):
    try:
        save_plot([], tmp_path / "empty.png")
    except ValueError as error:
        assert "At least one dataset result" in str(error)
    else:
        raise AssertionError("Expected ValueError when plotting no results")
