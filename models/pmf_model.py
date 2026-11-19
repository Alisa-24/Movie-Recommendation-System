
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from sklearn.model_selection import train_test_split

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
    """Calculate MSE on observed ratings with predictions clipped to 1–5."""

    squared_error = 0.0
    k = U.shape[1]

    for t in range(len(ratings)):
        u = user_idx[t]
        m = movie_idx[t]

        prediction = global_mean + user_bias[u] + movie_bias[m]

        for f in range(k):
            prediction += U[u, f] * V[m, f]

        prediction = min(max(prediction, 1.0), 5.0)
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
        self.validation_rmse_history = []
        self.best_epoch = None

    def fit(self, train, validation=None, *, patience=10):
        """Train PMF and restore the best checkpoint when given validation ratings."""

        if train.empty:
            raise ValueError("Training dataset is empty.")
        if validation is not None:
            if validation.empty:
                raise ValueError("Validation dataset is empty.")
            if self.epochs < 1 or patience < 1:
                raise ValueError("Early stopping requires positive epochs and patience.")

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
        self.validation_rmse_history = []
        self.best_epoch = None
        best_validation_rmse = np.inf
        best_checkpoint = None

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

            progress = (
                f"Epoch {epoch:02d}/{self.epochs} "
                f"| Training RMSE: {np.sqrt(mse):.4f}"
            )
            if validation is not None:
                validation_rmse = self.evaluate(validation)["PMF_RMSE"]
                self.validation_rmse_history.append(validation_rmse)
                progress += f" | Validation RMSE: {validation_rmse:.4f}"

                if validation_rmse < best_validation_rmse:
                    best_validation_rmse = validation_rmse
                    self.best_epoch = epoch
                    best_checkpoint = (
                        self.U.copy(), self.V.copy(),
                        self.user_bias.copy(), self.movie_bias.copy()
                    )

            print(progress)
            if validation is not None and epoch - (self.best_epoch or 0) >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if validation is not None:
            if best_checkpoint is None:
                raise ValueError("No finite validation RMSE was recorded.")
            self.U, self.V, self.user_bias, self.movie_bias = best_checkpoint
            print(
                f"Restored epoch {self.best_epoch} "
                f"| Validation RMSE: {best_validation_rmse:.4f}"
            )

        return self

    def predict_many(self, user_ids, movie_ids, *, clip=True):
        """Predict multiple ratings, with raw scores available for ranking."""

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

        return np.clip(predicted, 1, 5) if clip else predicted

    def predict(self, user_id, movie_id):
        """Predict a single user-movie rating."""

        return float(
            self.predict_many([user_id], [movie_id])[0]
        )

    def evaluate(self, test):
        """Evaluate held-out ratings using MSE and RMSE."""

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

    def save(self, metrics=None, *, validation_run=None):
        """Save factors and metrics, with the convergence trace used to select an epoch."""

        if self.U is None:
            raise ValueError("Model has not been trained.")

        convergence = self if validation_run is None else validation_run

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
            "epochs_trained": len(self.history),
            "random_state": self.random_state,
            "global_mean": self.global_mean
        }
        if convergence.best_epoch is not None:
            metadata["best_epoch"] = convergence.best_epoch
            metadata["best_validation_rmse"] = (
                convergence.validation_rmse_history[convergence.best_epoch - 1]
            )
            metadata["validation_epochs_trained"] = len(convergence.history)

        with open(factors_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Plot convergence
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            range(1, len(convergence.history) + 1),
            np.sqrt(convergence.history),
            marker="o",
            label="Training"
        )
        if convergence.best_epoch is not None:
            ax.plot(
                range(1, len(convergence.validation_rmse_history) + 1),
                convergence.validation_rmse_history,
                marker="o",
                label="Validation"
            )
            ax.axvline(
                convergence.best_epoch, linestyle="--",
                color="gray", label="Selected epoch"
            )
            ax.legend()
            ax.set_title("PMF Validation Run")
        else:
            ax.set_title("PMF Training Convergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            REPORTS_DIR / "pmf_convergence.png",
            dpi=150
        )
        plt.close(fig)

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
    # Reserve validation ratings from training and keep the shared SVD test split.
    train, test, _, _ = create_user_item_matrix()
    fit_rows, validation = train_test_split(train, test_size=0.1, random_state=42)

    validation_run = PMFModel(epochs=45)
    print("Selecting the PMF training epoch...")
    validation_run.fit(fit_rows, validation)

    print(f"Refitting PMF on all training ratings for {validation_run.best_epoch} epochs...")
    model = PMFModel(epochs=validation_run.best_epoch)
    model.fit(train)

    metrics = model.evaluate(test)

    print(f"PMF MSE: {metrics['PMF_MSE']:.4f}")
    print(f"PMF RMSE: {metrics['PMF_RMSE']:.4f}")

    model.save(metrics, validation_run=validation_run)

    print("PMF model completed successfully.")
