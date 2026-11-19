
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from utils.matrix_creation import create_user_item_matrix


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


class SVDModel:
    def __init__(self, k=50):
        self.k = k
        self.predictions = None
        self.user_ids = None
        self.movie_ids = None
        self.user_means = None
        self.global_mean = None

    def fit(self, matrix, user_means, global_mean):
        """Train SVD on the normalized user-item matrix."""

        if not 1 <= self.k < min(matrix.shape):
            raise ValueError("Invalid number of latent factors.")

        self.user_ids = matrix.index
        self.movie_ids = matrix.columns
        self.user_means = user_means.reindex(self.user_ids)
        self.global_mean = float(global_mean)

        # Convert to sparse matrix
        sparse_matrix = csr_matrix(
            matrix.to_numpy(dtype=np.float32)
        )

        # Perform SVD
        U, sigma, Vt = svds(
            sparse_matrix,
            k=self.k,
            rng=42
        )

        # Reconstruct predicted ratings
        self.predictions = (U * sigma) @ Vt

        # Restore original rating scale
        self.predictions += self.user_means.to_numpy()[:, None]

        # Restrict predictions to 1-5
        self.predictions = np.clip(
            self.predictions, 1, 5
        ).astype(np.float32)

        return self

    def predict(self, user_id, movie_id):
        """Predict a rating for a user and movie."""

        if self.predictions is None:
            raise ValueError("Model has not been trained.")

        user_idx = self.user_ids.get_indexer([user_id])[0]
        movie_idx = self.movie_ids.get_indexer([movie_id])[0]

        # Unseen user
        if user_idx == -1:
            return self.global_mean

        # Unseen movie
        if movie_idx == -1:
            return float(self.user_means.iloc[user_idx])

        return float(self.predictions[user_idx, movie_idx])

    def evaluate(self, test):
        """Calculate MSE and RMSE on test ratings."""

        if self.predictions is None:
            raise ValueError("Model has not been trained.")

        if test.empty:
            raise ValueError("Test dataset is empty.")

        user_idx = self.user_ids.get_indexer(test["UserID"])
        movie_idx = self.movie_ids.get_indexer(test["MovieID"])

        known_users = user_idx >= 0
        known_pairs = known_users & (movie_idx >= 0)

        predicted = np.full(
            len(test), self.global_mean, dtype=np.float32
        )

        # Use user mean for unseen movies
        predicted[known_users] = self.user_means.iloc[
            user_idx[known_users]
        ].to_numpy()

        # Use SVD predictions for known pairs
        predicted[known_pairs] = self.predictions[
            user_idx[known_pairs],
            movie_idx[known_pairs]
        ]

        actual = test["Rating"].to_numpy()

        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)

        return {
            "SVD_MSE": float(mse),
            "SVD_RMSE": float(rmse)
        }

    def save(self):
        """Save predictions and user/movie ID mappings."""

        if self.predictions is None:
            raise ValueError("Model has not been trained.")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        np.save(
            REPORTS_DIR / "svd_predictions.npy",
            self.predictions
        )
        np.save(
            REPORTS_DIR / "svd_user_ids.npy",
            self.user_ids.to_numpy()
        )
        np.save(
            REPORTS_DIR / "svd_movie_ids.npy",
            self.movie_ids.to_numpy()
        )


if __name__ == "__main__":
    train, test, matrix, user_means = create_user_item_matrix()

    model = SVDModel(k=50)

    print("Training SVD...")
    model.fit(
        matrix,
        user_means,
        global_mean=train["Rating"].mean()
    )

    metrics = model.evaluate(test)

    print("SVD MSE:", metrics["SVD_MSE"])
    print("SVD RMSE:", metrics["SVD_RMSE"])

    model.save()

    # Preserve existing metrics from other models
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "model_metrics.json"

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics.update(metrics)

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("SVD model completed successfully.")
