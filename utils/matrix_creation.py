
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"


def normalize_matrix(matrix):
    """Normalize ratings by subtracting each user's mean rating."""

    # Calculate the average rating for each user
    user_means = matrix.mean(axis=1)

    # Subtract each user's mean from their ratings
    normalized_matrix = matrix.sub(user_means, axis=0)

    # Replace missing ratings with 0 after normalization
    normalized_matrix = normalized_matrix.fillna(0)

    return normalized_matrix, user_means


def create_user_item_matrix():
    """Split ratings and create the normalized user-item matrix."""

    ratings, movies, users = load_data()

    # Split into training and testing sets
    train, test = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42
    )

    # Create user-item matrix from training data only
    matrix = train.pivot(
        index="UserID",
        columns="MovieID",
        values="Rating"
    )

    # Normalize the matrix
    normalized_matrix, user_means = normalize_matrix(matrix)

    # Save normalized matrix
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalized_matrix.round(3).to_csv(
        PROCESSED_DIR / "user_item_matrix.csv"
    )

    return train, test, normalized_matrix, user_means


if __name__ == "__main__":
    train, test, matrix, user_means = create_user_item_matrix()

    print("Training shape:", train.shape)
    print("Testing shape:", test.shape)
    print("\nNormalized matrix:")
    print(matrix.iloc[:5, :10])
    print("\nMatrix shape:", matrix.shape)
    print("\nUser means:")
    print(user_means.head())
