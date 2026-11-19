
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    """Load and clean MovieLens datasets."""

    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )

    movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )

    users = pd.read_csv(
        DATA_DIR / "users.dat",
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
        dtype={"ZipCode": "string"}
    )

    # Remove null values and exact duplicates
    ratings = ratings.dropna().drop_duplicates()
    movies = movies.dropna().drop_duplicates()
    users = users.dropna().drop_duplicates()

    # Keep the latest rating for each user-movie pair
    ratings = (
        ratings.sort_values("Timestamp")
        .drop_duplicates(subset=["UserID", "MovieID"], keep="last")
        .sort_values(["UserID", "MovieID"])
        .reset_index(drop=True)
    )

    movies = movies.reset_index(drop=True)
    users = users.reset_index(drop=True)

    return ratings, movies, users


if __name__ == "__main__":
    ratings, movies, users = load_data()

    for name, df in [
        ("Ratings", ratings),
        ("Movies", movies),
        ("Users", users)
    ]:
        print(f"\n{name} Dataset:")
        print(df.head())
        print(f"Shape: {df.shape}")
        print(f"Null values: {df.isnull().sum().sum()}")
        print(f"Duplicates: {df.duplicated().sum()}")
