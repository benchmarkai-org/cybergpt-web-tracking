import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Union, Tuple, Any
from sklearn.metrics import silhouette_score
import pandas as pd
from tqdm.auto import tqdm

RANDOM_STATE = 42


def _aggregate_vectors(vectors: np.ndarray, aggregations: List[str]) -> np.ndarray:
    """Aggregate vectors using specified statistical measures."""
    agg_funcs = {
        "mean": np.mean,
        "std": np.std,
        "median": np.median,
        "q1": lambda x: np.percentile(x, 25, axis=0),
        "q3": lambda x: np.percentile(x, 75, axis=0),
    }
    return np.concatenate([agg_funcs[agg](vectors, axis=0) for agg in aggregations])


def create_user_representations(
    user_data: Dict[str, List[np.ndarray]],
    aggregations: List[str] = ["mean", "std", "median", "q1", "q3"],
) -> Dict[str, np.ndarray]:
    """Create user representations by aggregating their vectors."""
    representations = {}
    for user_id, vectors in user_data.items():
        representations[user_id] = _aggregate_vectors(vectors, aggregations)
    return representations


def _get_clustering_algorithms(
    n_clusters: int, random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """Get clustering algorithms."""
    return {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=random_state),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
    }


def cluster_users(
    representations: Dict[str, np.ndarray],
    algorithm_names: List[str] = ["kmeans", "agglomerative"],
    n_clusters: int = 5,
    return_scores: bool = False,
    random_state: int = RANDOM_STATE,
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, float]]]:
    """Cluster users based on their vector representations."""
    X = np.array(list(representations.values()))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    scores = {}
    algorithms = _get_clustering_algorithms(n_clusters, random_state)
    for name in algorithm_names:
        labels = algorithms[name].fit_predict(X_scaled)
        results[name] = labels
        scores[name] = silhouette_score(X_scaled, labels)

    if return_scores:
        return results, scores
    return results


def compute_cluster_scores(
    X: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute silhouette scores for n_clusters between min_clusters and max_clusters."""
    results = []
    for k in tqdm(
        range(min_clusters, max_clusters + 1), desc="Computing cluster scores"
    ):
        _, scores = cluster_users(
            X, n_clusters=k, return_scores=True, random_state=random_state
        )
        results.extend(
            [
                {"n_clusters": k, "algorithm": algo, "score": score}
                for algo, score in scores.items()
            ]
        )
    return pd.DataFrame(results)
