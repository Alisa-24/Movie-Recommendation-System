
import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.svd_model import SVDModel
from models.pmf_model import PMFModel
from utils.data_loader import load_data


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@lru_cache(maxsize=1)
def get_movie_data():
    """Load movie metadata and rating history."""
    ratings, movies, _ = load_data()
    return ratings, movies


def load_saved_models():
    """Load previously trained SVD and PMF models."""

    # Load SVD model
    svd = SVDModel()

    svd.predictions = np.load(
        REPORTS_DIR / "svd_predictions.npy",
        mmap_mode="r"
    )

    svd.user_ids = pd.Index(
        np.load(REPORTS_DIR / "svd_user_ids.npy")
    )

    svd.movie_ids = pd.Index(
        np.load(REPORTS_DIR / "svd_movie_ids.npy")
    )

    # Load PMF model
    factors_dir = REPORTS_DIR / "pmf_factors"

    with open(factors_dir / "metadata.json") as f:
        metadata = json.load(f)

    pmf = PMFModel(k=metadata["k"])

    pmf.U = np.load(factors_dir / "U.npy")
    pmf.V = np.load(factors_dir / "V.npy")

    pmf.user_bias = np.load(
        factors_dir / "user_bias.npy"
    )

    pmf.movie_bias = np.load(
        factors_dir / "movie_bias.npy"
    )

    pmf.user_ids = pd.Index(
        np.load(factors_dir / "user_ids.npy")
    )

    pmf.movie_ids = pd.Index(
        np.load(factors_dir / "movie_ids.npy")
    )

    pmf.global_mean = metadata["global_mean"]

    return svd, pmf


def generate_recommendations(user_id, model, top_n=10):
    """Generate top-N unseen movie recommendations for a user."""

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    if user_id not in model.user_ids:
        raise ValueError(f"User {user_id} is unknown to the model.")

    ratings, movies = get_movie_data()

    # Movies already rated by the user
    rated_movies = ratings.loc[
        ratings["UserID"] == user_id, "MovieID"
    ]

    # Select unseen movies known to the model
    candidates = movies[
        movies["MovieID"].isin(model.movie_ids)
        & ~movies["MovieID"].isin(rated_movies)
    ].copy()

    columns = [
        "UserID", "Model", "MovieID",
        "Title", "Genres", "PredictedRating"
    ]

    if candidates.empty:
        return pd.DataFrame(columns=columns)

    movie_ids = candidates["MovieID"].to_numpy()

    # Generate predictions
    if isinstance(model, SVDModel):

        user_idx = model.user_ids.get_loc(user_id)
        movie_idx = model.movie_ids.get_indexer(movie_ids)

        predictions = model.predictions[
            user_idx, movie_idx
        ]

        model_name = "SVD"

    elif isinstance(model, PMFModel):

        user_ids = np.full(len(movie_ids), user_id)

        predictions = model.predict_many(
            user_ids, movie_ids, clip=False
        )

        model_name = "PMF"

    else:
        raise TypeError("Unsupported recommendation model.")

    candidates["PredictedRating"] = predictions

    # Sort by predicted rating
    recommendations = (
        candidates.nlargest(top_n, "PredictedRating")
        .copy()
    )

    display_ratings = recommendations["PredictedRating"]
    if model_name == "PMF":
        display_ratings = display_ratings.clip(1, 5)
    recommendations["PredictedRating"] = display_ratings.round(3)

    recommendations["UserID"] = user_id
    recommendations["Model"] = model_name

    return recommendations[columns].reset_index(drop=True)


def _shorten_title(title, max_length=30):
    """Shorten movie titles for plots."""

    title = str(title)

    if len(title) <= max_length:
        return title

    return title[:max_length - 3] + "..."


def plot_user_comparison(user_id, svd, pmf, top_n=10):
    """Compare actual, SVD and PMF ratings on held-out test movies."""

    ratings, movies = get_movie_data()

    # Reproduce the same train/test split
    _, test = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42
    )

    # Select test ratings for this user
    user_test = test[
        test["UserID"] == user_id
    ].copy()

    # Keep movies known to both models
    user_test = user_test[
        user_test["MovieID"].isin(svd.movie_ids)
        & user_test["MovieID"].isin(pmf.movie_ids)
    ]

    if user_id not in svd.user_ids or user_id not in pmf.user_ids:
        print(f"User {user_id} is unknown to one of the models.")
        return

    if user_test.empty:
        print(f"No test ratings found for user {user_id}.")
        return
    
    # Evaluate all test ratings for this user
    all_movie_ids = user_test["MovieID"].to_numpy()
    actual_all = user_test["Rating"].to_numpy()

    # Evaluate predictions for all test movies
    svd_user_idx = svd.user_ids.get_loc(user_id)
    svd_movie_idx = svd.movie_ids.get_indexer(all_movie_ids)

    svd_all = svd.predictions[svd_user_idx, svd_movie_idx]

    pmf_all = pmf.predict_many(
        np.full(len(all_movie_ids), user_id),
        all_movie_ids
    )

    svd_rmse = np.sqrt(np.mean((actual_all - svd_all) ** 2))
    pmf_rmse = np.sqrt(np.mean((actual_all - pmf_all) ** 2))

    print(f"User {user_id} SVD RMSE: {svd_rmse:.4f}")
    print(f"User {user_id} PMF RMSE: {pmf_rmse:.4f}")

    # Sample up to top_n movies
    user_test = user_test.sample(
        n=min(top_n, len(user_test)),
        random_state=42
    )

    # Add movie titles
    user_test = user_test.merge(
        movies[["MovieID", "Title"]],
        on="MovieID",
        how="left"
    )

    movie_ids = user_test["MovieID"].to_numpy()

    # Actual ratings
    actual = user_test["Rating"].to_numpy()

    # SVD predictions
    user_idx = svd.user_ids.get_loc(user_id)
    movie_idx = svd.movie_ids.get_indexer(movie_ids)

    svd_preds = svd.predictions[
        user_idx, movie_idx
    ]

    # PMF predictions
    pmf_preds = pmf.predict_many(
        np.full(len(movie_ids), user_id),
        movie_ids
    )

    # Plot actual vs predicted ratings
    x = np.arange(len(movie_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(
        x - width, actual, width,
        label="Actual"
    )

    ax.bar(
        x, svd_preds, width,
        label="SVD"
    )

    ax.bar(
        x + width, pmf_preds, width,
        label="PMF"
    )

    labels = [
        _shorten_title(title)
        for title in user_test["Title"]
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_xlabel("Movies")
    ax.set_ylabel("Rating")
    ax.set_title(
        f"Actual vs SVD vs PMF - User {user_id}\n"
        f"User Test RMSE | SVD: {svd_rmse:.3f} | PMF: {pmf_rmse:.3f}"
    )

    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR / "user_comparison.png",
        dpi=150
    )

    plt.close(fig)


def plot_top_recommendations(user_id, svd_recs, pmf_recs):
    """Plot top recommendations from SVD and PMF."""

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 10)
    )

    models = [
        ("SVD", svd_recs),
        ("PMF", pmf_recs)
    ]

    for ax, (model_name, recommendations) in zip(axes, models):

        plot_data = recommendations.sort_values(
            "PredictedRating",
            ascending=True
        )

        labels = [
            _shorten_title(title)
            for title in plot_data["Title"]
        ]

        ax.barh(
            labels,
            plot_data["PredictedRating"]
        )

        ax.set_title(
            f"Top {len(plot_data)} {model_name} "
            f"Recommendations - User {user_id}"
        )

        ax.set_xlabel("Predicted Rating")
        ax.set_ylabel("Movie Title")
        ax.set_xlim(0, 5.2)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR / "top_recommendations.png",
        dpi=150
    )

    plt.close(fig)


def compare_recommendations(user_id, svd, pmf, top_n=10):
    """Generate, compare and save recommendations."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate recommendations
    svd_recs = generate_recommendations(
        user_id, svd, top_n
    )

    pmf_recs = generate_recommendations(
        user_id, pmf, top_n
    )

    # Combine recommendations
    combined = pd.concat(
        [svd_recs, pmf_recs],
        ignore_index=True
    )

    # Save recommendations to CSV
    combined.to_csv(
        REPORTS_DIR / f"user_{user_id}_recommendations.csv",
        index=False
    )

    # Plot actual vs predicted ratings
    plot_user_comparison(
        user_id, svd, pmf, top_n
    )

    # Plot top recommendations
    plot_top_recommendations(
        user_id, svd_recs, pmf_recs
    )

    return svd_recs, pmf_recs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--user-id",
        type=int,
        default=1
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10
    )

    args = parser.parse_args()

    # Load trained models
    svd, pmf = load_saved_models()

    # Generate and compare
    svd_recs, pmf_recs = compare_recommendations(
        args.user_id,
        svd,
        pmf,
        args.top_n
    )

    print(f"\nTop {args.top_n} SVD Recommendations:")
    print(svd_recs.to_string(index=False))

    print(f"\nTop {args.top_n} PMF Recommendations:")
    print(pmf_recs.to_string(index=False))

    print("\nRecommendations and plots saved successfully.")
