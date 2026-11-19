
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

from utils.matrix_creation import create_user_item_matrix


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@njit(cache=True)
def _sgd_epoch(
    user_idx, movie_idx, ratings, order,
    U, V, user_bias, movie_bias,
    global_mean, learning_rate, reg
):
    """Perform one SGD training epoch."""

    k = U.shape[1]

    for t in order:
        u = user_idx[t]
        m = movie_idx[t]

        # Predict the rating
        prediction = global_mean + user_bias[u] + movie_bias[m]

        for f in range(k):
            prediction += U[u, f] * V[m, f]

        error = ratings[t] - prediction

        # Update user and movie biases
        user_bias[u] += learning_rate * (
            error - reg * user_bias[u]
        )
        movie_bias[m] += learning_rate * (
            error - reg * movie_bias[m]
        )

        # Update latent factors
        for f in range(k):
            old_u = U[u, f]
            old_v = V[m, f]

            U[u, f] += learning_rate * (
                error * old_v - reg * old_u
            )
            V[m, f] += learning_rate * (
                error * old_u - reg * old_v
            )


@njit(cache=True)
def _training_mse(
    user_idx, movie_idx, ratings,
    U, V, user_bias, movie_bias, global_mean
):
    """Calculate training MSE using observed ratings only."""

    squared_error = 0.0
    k = U.shape[1]

    for t in range(len(ratings)):
        u = user_idx[t]
        m = movie_idx[t]

        prediction = global_mean + user_bias[u] + movie_bias[m]

        for f in range(k):
            prediction += U[u, f] * V[m, f]

        error = ratings[t] - prediction
        squared_error += error ** 2

    return squared_error / len(ratings)


class PMFModel:
    def __init__(
        self,
        k=50,
        learning_rate=0.01,
        reg=0.05,
        epochs=25,
        random_state=42
    ):
        self.k = k
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state

        self.U = None
        self.V = None
        self.user_bias = None
        self.movie_bias = None

        self.user_ids = None
        self.movie_ids = None
        self.global_mean = None

        self.history = []

    def fit(self, train):
        """Train PMF using observed user-movie ratings."""

        if train.empty:
            raise ValueError("Training dataset is empty.")

        # Create user/movie ID mappings
        self.user_ids = pd.Index(
            np.sort(train["UserID"].unique())
        )
        self.movie_ids = pd.Index(
            np.sort(train["MovieID"].unique())
        )

        user_idx = self.user_ids.get_indexer(
            train["UserID"]
        ).astype(np.int32)

        movie_idx = self.movie_ids.get_indexer(
            train["MovieID"]
        ).astype(np.int32)

        ratings = train["Rating"].to_numpy(
            dtype=np.float32
        )

        self.global_mean = float(ratings.mean())

        # Initialize latent factors randomly
        rng = np.random.default_rng(self.random_state)

        self.U = rng.normal(
            0, 0.1, (len(self.user_ids), self.k)
        ).astype(np.float32)

        self.V = rng.normal(
            0, 0.1, (len(self.movie_ids), self.k)
        ).astype(np.float32)

        self.user_bias = np.zeros(
            len(self.user_ids), dtype=np.float32
        )
        self.movie_bias = np.zeros(
            len(self.movie_ids), dtype=np.float32
        )

        self.history = []

        # Train using SGD
        for epoch in range(1, self.epochs + 1):

            order = rng.permutation(
                len(ratings)
            ).astype(np.int32)

            _sgd_epoch(
                user_idx, movie_idx, ratings, order,
                self.U, self.V,
                self.user_bias, self.movie_bias,
                self.global_mean,
                self.learning_rate, self.reg
            )

            mse = float(_training_mse(
                user_idx, movie_idx, ratings,
                self.U, self.V,
                self.user_bias, self.movie_bias,
                self.global_mean
            ))

            self.history.append(mse)

            print(
                f"Epoch {epoch:02d}/{self.epochs} "
                f"| Training MSE: {mse:.4f}"
            )

        return self

    def predict_many(self, user_ids, movie_ids):
        """Predict ratings for multiple user-movie pairs."""

        if self.U is None:
            raise ValueError("Model has not been trained.")

        user_idx = self.user_ids.get_indexer(user_ids)
        movie_idx = self.movie_ids.get_indexer(movie_ids)

        predicted = np.full(
            len(user_idx),
            self.global_mean,
            dtype=np.float32
        )

        known_users = user_idx >= 0
        known_movies = movie_idx >= 0

        predicted[known_users] += self.user_bias[
            user_idx[known_users]
        ]
        predicted[known_movies] += self.movie_bias[
            movie_idx[known_movies]
        ]

        known_pairs = known_users & known_movies

        predicted[known_pairs] += np.einsum(
            "ij,ij->i",
            self.U[user_idx[known_pairs]],
            self.V[movie_idx[known_pairs]]
        )

        return np.clip(predicted, 1, 5)

    def predict(self, user_id, movie_id):
        """Predict a single user-movie rating."""

        return float(
            self.predict_many([user_id], [movie_id])[0]
        )

    def evaluate(self, test):
        """Evaluate PMF using test-set MSE and RMSE."""

        if test.empty:
            raise ValueError("Test dataset is empty.")

        predictions = self.predict_many(
            test["UserID"], test["MovieID"]
        )

        actual = test["Rating"].to_numpy(
            dtype=np.float32
        )

        mse = float(np.mean((actual - predictions) ** 2))
        rmse = float(np.sqrt(mse))

        return {
            "PMF_MSE": mse,
            "PMF_RMSE": rmse
        }

    def save(self, metrics=None):
        """Save latent factors, convergence plot and metrics."""

        if self.U is None:
            raise ValueError("Model has not been trained.")

        factors_dir = REPORTS_DIR / "pmf_factors"
        factors_dir.mkdir(parents=True, exist_ok=True)

        np.save(factors_dir / "U.npy", self.U)
        np.save(factors_dir / "V.npy", self.V)
        np.save(factors_dir / "user_bias.npy", self.user_bias)
        np.save(factors_dir / "movie_bias.npy", self.movie_bias)
        np.save(
            factors_dir / "user_ids.npy",
            self.user_ids.to_numpy()
        )
        np.save(
            factors_dir / "movie_ids.npy",
            self.movie_ids.to_numpy()
        )

        metadata = {
            "k": self.k,
            "learning_rate": self.learning_rate,
            "reg": self.reg,
            "epochs": self.epochs,
            "random_state": self.random_state,
            "global_mean": self.global_mean
        }

        with open(factors_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Plot convergence
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(self.history) + 1),
            self.history,
            marker="o"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Training MSE")
        plt.title("PMF Training Convergence")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            REPORTS_DIR / "pmf_convergence.png",
            dpi=150
        )
        plt.close()

        # Preserve existing SVD metrics
        if metrics is not None:
            metrics_path = REPORTS_DIR / "model_metrics.json"

            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)

            if "SVD_RMSE" in all_metrics:
                svd_rmse = all_metrics["SVD_RMSE"]
                pmf_rmse = all_metrics["PMF_RMSE"]

                all_metrics["PMF_vs_SVD_improvement_%"] = (
                    (svd_rmse - pmf_rmse) / svd_rmse * 100
                )

            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=4)


if __name__ == "__main__":
    # Reuse the exact same train/test split as SVD
    train, test, _, _ = create_user_item_matrix()

    model = PMFModel(
        k=50,
        learning_rate=0.01,
        reg=0.05,
        epochs=25,
        random_state=42
    )

    print("Training PMF...")
    model.fit(train)

    metrics = model.evaluate(test)

    print(f"PMF MSE: {metrics['PMF_MSE']:.4f}")
    print(f"PMF RMSE: {metrics['PMF_RMSE']:.4f}")

    model.save(metrics)

    print("PMF model completed successfully.")
