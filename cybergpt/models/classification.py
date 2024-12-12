import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class PCAReducer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None

    def fit(self, X_train, n_components):
        """Fit PCA on the training set."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)

    def reduce_dimensions(self, X):
        """Reduce dimensions of given dataset."""
        if self.pca is None:
            raise ValueError("Run fit first!")

        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)

        print(f"Original shape: {X.shape}")
        print(f"Reduced shape: {X_reduced.shape}")
        print(
            f"Total explained variance ratio: {sum(self.pca.explained_variance_ratio_):.3f}"
        )

        return X_reduced

    def analyze_explained_variance(self):
        cumulative_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(cumulative_variance_ratio) + 1),
            cumulative_variance_ratio,
            "bo-",
        )
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("Explained Variance vs Number of Components")
        plt.grid(True)
        plt.show()

        return cumulative_variance_ratio


class ClassificationEvaluation:
    def __init__(self, X, y, train_indices=None, cv_splits=None):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

        self.train_indices = train_indices
        self.cv_splits = cv_splits

    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data and scale features."""
        if self.train_indices is None:
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                self.train_indices,
                _,
            ) = train_test_split(
                self.X,
                self.y,
                np.arange(self.X.shape[0]),
                test_size=test_size,
                random_state=random_state,
                stratify=self.y,
            )
        else:
            test_indices = np.setdiff1d(np.arange(self.X.shape[0]), self.train_indices)
            self.X_train = self.X[self.train_indices, :]
            self.X_test = self.X[test_indices, :]
            self.y_train = self.y[self.train_indices]
            self.y_test = self.y[test_indices]

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def evaluate_models(self, model_names=None):
        """Evaluate multiple models and return their results."""
        models = {
            "logistic": LogisticRegression(max_iter=1000, multi_class="multinomial"),
            # 'knn': KNeighborsClassifier(n_neighbors=5),
            "rf": RandomForestClassifier(n_estimators=100),
        }
        if model_names is not None:
            models = {name: models[name] for name in model_names}

        results = {}

        if self.cv_splits is None:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            self.cv_splits = list(skf.split(self.X_train, self.y_train))

        for name, model in models.items():
            print(f"\nEvaluating {name}...")

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            cv_scores = np.array(
                [
                    model.fit(self.X_train[train_idx], self.y_train[train_idx]).score(
                        self.X_train[val_idx], self.y_train[val_idx]
                    )
                    for train_idx, val_idx in self.cv_splits
                ]
            )

            results[name] = {
                "model": model,
                "classification_report": classification_report(
                    self.y_test, y_pred, output_dict=True
                ),
                "confusion_matrix": confusion_matrix(self.y_test, y_pred),
                "cv_scores": cv_scores,
            }

            print(f"\nClassification Report for {name}:")
            clf_report_summary = results[name]["classification_report"]
            clf_report_summary = {
                k: v
                for k, v in clf_report_summary.items()
                if k in ["accuracy", "macro avg", "weighted avg"]
            }
            for k, v in clf_report_summary.items():
                print(f"{k}: {v}")
            print(
                f"\nCross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
            )

        return results

    def plot_confusion_matrix(self, confusion_matrix, title):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, cmap="Blues")
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
