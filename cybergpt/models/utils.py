import pandas as pd
import numpy as np
from scipy.special import digamma, gammaln


def feature_df_to_numpy(features_df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to numpy array with one-hot encoding for categorical variables."""
    df = features_df.copy()

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) > 0:
        encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    else:
        encoded = df

    return encoded.values


def fit_dirichlet(
    vectors: np.ndarray, max_iter: int = 1000, tol: float = 1e-7
) -> np.ndarray:
    """
    Find MLE of Dirichlet parameters using fixed-point iteration
    Based on Thomas Minka's fixed-point method.
    """
    # Initialize alpha with method of moments estimate
    alpha = np.mean(vectors, axis=0) * vectors.shape[1]

    # Ensure alpha is positive and not too close to zero
    alpha = np.maximum(alpha, 0.001)

    for _ in range(max_iter):
        alpha_old = alpha.copy()

        # Fixed point update
        alpha = alpha * (
            digamma(np.sum(alpha)) - digamma(alpha) + np.mean(np.log(vectors), axis=0)
        )

        # Ensure numerical stability
        alpha = np.maximum(alpha, 0.001)

        # Check convergence
        if np.max(np.abs(alpha - alpha_old)) < tol:
            break

    return alpha


def dirichlet_kl_divergence(alpha, beta):
    """
    Compute KL divergence between two Dirichlet distributions
    """
    sum_alpha = np.sum(alpha)
    sum_beta = np.sum(beta)

    term1 = gammaln(sum_alpha) - gammaln(sum_beta)
    term2 = -np.sum(gammaln(alpha) - gammaln(beta))
    term3 = np.sum((alpha - beta) * (digamma(alpha) - digamma(sum_alpha)))

    return term1 + term2 + term3


def dirichlet_js_matrix(dirichlet_params: np.ndarray) -> np.ndarray:
    """
    Compute matrix of pairwise Jensen-Shannon divergences between Dirichlet distributions
    """
    L = len(dirichlet_params)
    distances = np.zeros((L, L))

    for i in range(L):
        for j in range(i + 1, L):
            M = (dirichlet_params[i] + dirichlet_params[j]) / 2

            kl1 = dirichlet_kl_divergence(dirichlet_params[i], M)
            kl2 = dirichlet_kl_divergence(dirichlet_params[j], M)
            jsd = (kl1 + kl2) / 2

            distances[i, j] = jsd
            distances[j, i] = jsd

    return distances
