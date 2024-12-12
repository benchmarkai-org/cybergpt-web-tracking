import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import digamma, gammaln
from tqdm.auto import tqdm
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_dirichlet(
    vectors: np.ndarray, max_iter: int = 1000, tol: float = 1e-7
) -> np.ndarray:
    """
    Find MLE of Dirichlet parameters using fixed-point iteration
    Based on Thomas Minka's fixed-point method
    """
    eps = 0.001
    alpha = np.mean(vectors, axis=0) * vectors.shape[1]
    alpha = np.maximum(alpha, eps)  # Ensure alpha is not too close to zero

    for _ in range(max_iter):
        alpha_old = alpha.copy()
        # Fixed point update
        alpha = alpha * (
            digamma(np.sum(alpha)) - digamma(alpha) + np.mean(np.log(vectors), axis=0)
        )
        alpha = np.maximum(alpha, eps)

        if np.max(np.abs(alpha - alpha_old)) < tol:
            break

    return alpha


def project_dirichlet_by_class(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    n_supercomponents: int,
    reg_covar: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects high-dimensional embeddings onto a lower-dimensional Dirichlet distribution space
    for each unique class label. This is done by first fitting a Gaussian Mixture Model (GMM)
    to get probability distributions, then fitting a Dirichlet distribution to these
    probabilities for each class.

    Args:
        embeddings (np.ndarray): Input embeddings matrix of shape (N, D) where:
            - N is the number of samples
            - D is the dimensionality of the embedding space
        labels (np.ndarray): Array of class labels of shape (N,) containing L unique labels
        n_components (int): Number of GMM components (M) to use for the projection
        n_supercomponents (int): Number of PCA components to use for the projection
        reg_covar (float): Regularization parameter for the covariance matrix of the GMM

    Returns:
        np.ndarray: Matrix of Dirichlet parameters of shape (L, M) where:
            - L is the number of unique labels
            - M is the number of mixture components
            Each row contains the Dirichlet parameters for one class label.
        np.ndarray: Matrix of GMM probabilities of shape (N, M) where:
            - N is the number of samples
            - M is the number of mixture components
            Each row contains the probability distribution for one sample.

    Notes:
        The process involves:
        1. Reducing the dimensionality of the embeddings using PCA
        2. Fitting a GMM with M components to the embeddings
        3. Converting embeddings to probability distributions using GMM
        4. Fitting a Dirichlet distribution to the probabilities for each class
    """
    print(
        f"Reducing dimensionality from {embeddings.shape[1]} to {n_supercomponents}..."
    )
    X = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=min(X.shape[1], n_supercomponents))
    X_reduced = pca.fit_transform(X)

    print(f"Fitting GMM with {n_components} components...")
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42,
        verbose=1,
        reg_covar=reg_covar,
        covariance_type="full",
    )
    gmm.fit(X_reduced)

    print("Computing GMM probabilities...")
    probs = gmm.predict_proba(X_reduced)

    unique_labels = np.unique(labels)
    label_dirichlet_params = np.zeros((len(unique_labels), n_components))

    print(f"Fitting Dirichlet distributions for {len(unique_labels)} classes...")
    for i, label in enumerate(tqdm(unique_labels, desc="Processing classes")):
        label_mask = labels == label
        label_probs = probs[label_mask]
        label_dirichlet_params[i] = fit_dirichlet(label_probs)

    return probs, label_dirichlet_params


def dirichlet_kl_divergence(alpha, beta):
    """
    Compute KL divergence between two Dirichlet distributions
    """
    sum_alpha = np.sum(alpha)
    sum_beta = np.sum(beta)

    log_gamma_diff = gammaln(sum_alpha) - gammaln(sum_beta)
    log_gamma_correction = -np.sum(gammaln(alpha) - gammaln(beta))
    digamma_expectation = np.sum((alpha - beta) * (digamma(alpha) - digamma(sum_alpha)))

    return log_gamma_diff + log_gamma_correction + digamma_expectation


def dirichlet_js_matrix(dirichlet_params: np.ndarray) -> np.ndarray:
    """
    Compute matrix of pairwise Jensen-Shannon divergences between Dirichlet distributions
    """
    L = len(dirichlet_params)
    distances = np.zeros((L, L))

    for i in range(L):
        for j in range(i + 1, L):
            # Midpoint mixture
            mixture = (dirichlet_params[i] + dirichlet_params[j]) / 2

            kl1 = dirichlet_kl_divergence(dirichlet_params[i], mixture)
            kl2 = dirichlet_kl_divergence(dirichlet_params[j], mixture)
            jsd = (kl1 + kl2) / 2

            distances[i, j] = jsd
            distances[j, i] = jsd

    return distances


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate synthetic data
    np.random.seed(42)

    # Parameters
    N = 10000  # Number of samples
    D = 500  # Embedding dimension
    L = 200  # Number of unique labels
    M = 50  # Number of mixture components

    embeddings = []
    labels = []

    for label in range(L):
        mean = np.random.randn(D) * (label + 1)
        samples = np.random.randn(N // L, D) + mean
        embeddings.append(samples)
        labels.extend([label] * (N // L))

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # Project embeddings to Dirichlet space
    dirichlet_params = project_dirichlet_by_class(embeddings, labels, n_components=M)

    print("Shape of Dirichlet parameters:", dirichlet_params.shape)
    print("\nDirichlet parameters for each class:")
    for i in range(L):
        print(f"Class {i}:", dirichlet_params[i])

    # Compute Jensen-Shannon divergence matrix
    js_matrix = dirichlet_js_matrix(dirichlet_params)
    print("Shape of Jensen-Shannon matrix:", js_matrix.shape)

    plt.imshow(js_matrix, cmap="viridis")
    plt.colorbar()
    plt.show()
