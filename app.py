from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import DATA_DIR
from utils.matrix_creation import split_ratings
from utils.recommendation import (
    REPORTS_DIR,
    generate_recommendations,
    get_movie_data,
    get_user_test_predictions,
    load_saved_models,
)


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_COMMANDS = "python -m models.svd_model\npython -m models.pmf_model"
REPORT_COMMAND = "python -m utils.recommendation --user-id 1"
RATING_SERIES = ["Actual", "SVD", "PMF"]


@st.cache_resource(show_spinner="Loading MovieLens ratings and trained models…")
def load_dashboard_data():
    ratings, movies = get_movie_data()
    svd, pmf = load_saved_models()
    _, test = split_ratings(ratings)
    movies = movies.set_index("MovieID")
    return ratings, movies, test, svd, pmf


def show_movie_table(movies, rating_column, rating_label):
    table = movies[["Title", "Genres", rating_column]].reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    table["Genres"] = table["Genres"].str.replace("|", ", ", regex=False)
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Title": st.column_config.TextColumn("Movie", width="medium"),
            rating_column: st.column_config.NumberColumn(rating_label, format="%.2f"),
        },
    )


def show_comparison(predictions, titles):
    if predictions.empty:
        st.info("This user has no held-out ratings for movies both models can score.")
        return

    squared_errors = predictions[["SVD", "PMF"]].sub(predictions["Rating"], axis=0) ** 2
    rmse = np.sqrt(squared_errors.mean())
    svd_metric, pmf_metric, count = st.columns(3)
    svd_metric.metric("SVD RMSE", f"{rmse['SVD']:.3f}", border=True)
    pmf_metric.metric("PMF RMSE", f"{rmse['PMF']:.3f}", border=True)
    count.metric("Comparable test ratings", len(predictions), border=True)

    sample = predictions.sample(n=min(10, len(predictions)), random_state=42)
    sample = sample.rename(columns={"Rating": "Actual"})
    sample["Movie"] = sample["MovieID"].map(titles).fillna("Movie " + sample["MovieID"].astype(str))
    duplicate_titles = sample["Movie"].duplicated(keep=False)
    sample.loc[duplicate_titles, "Movie"] += " #" + sample.loc[duplicate_titles, "MovieID"].astype(str)
    chart_data = sample.melt(
        id_vars="Movie", value_vars=RATING_SERIES, var_name="Series", value_name="Rating"
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar(size=8, cornerRadiusEnd=2)
        .encode(
            y=alt.Y("Movie:N", title=None, sort=sample["Movie"].tolist(), axis=alt.Axis(labelLimit=200)),
            yOffset=alt.YOffset("Series:N", sort=RATING_SERIES),
            x=alt.X("Rating:Q", title="Rating", scale=alt.Scale(domain=[0, 5]), axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=RATING_SERIES, range=["#64748b", "#0d9488", "#cb690e"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["Movie:N", "Series:N", alt.Tooltip("Rating:Q", format=".2f")],
        )
        .properties(height=alt.Step(42))
    )
    st.subheader("Predicted vs. actual ratings")
    st.caption(f"Showing {len(sample)} of {len(predictions)} comparable test ratings.")
    st.altair_chart(
        chart,
        width="stretch",
        alt="Actual ratings and predictions from SVD and PMF for the same held-out movies.",
    )


def show_model_reports():
    """Show the convergence and aggregate test charts from the training run."""
    for filename in (
        "pmf_convergence.png",
        "rmse_comparison.png",
        "predicted_vs_actual.png",
    ):
        path = REPORTS_DIR / filename
        if path.is_file():
            st.image(str(path))
        else:
            st.info(f"Missing report: {filename}. Train both models and export the reports.")
            st.code(f"{TRAIN_COMMANDS}\n{REPORT_COMMAND}", language="bash")
            break


def main():
    st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")
    st.html(
        """
        <style>
          .movielens-header {
            padding: clamp(1.3rem, 3vw, 2.5rem);
            border-radius: 18px;
            border-left: 5px solid #5eead4;
            background: radial-gradient(ellipse at 95% 0%, #285164, transparent 62%), #102838;
            color: #f0f9fb;
          }
          .movielens-header h1 {
            color: inherit;
            letter-spacing: -0.045em;
            font-size: clamp(2rem, 4.4vw, 3.6rem);
            line-height: 1.06;
            padding: 0;
            margin: 0 0 0.55rem;
          }
          .movielens-header p {
            margin: 0;
            color: #c9e3e9;
          }
        </style>
        <header class="movielens-header">
          <h1>MovieLens Recommender</h1>
          <p>Choose a user to compare movie picks from SVD and PMF.</p>
        </header>
        """
    )

    try:
        ratings, movies, test, svd, pmf = load_dashboard_data()
    except FileNotFoundError as error:
        missing = Path(error.filename)
        if missing.is_relative_to(DATA_DIR):
            st.error(f"Missing MovieLens file: data/{missing.name}")
            st.markdown("Place all three MovieLens 1M data files in the `data/` folder.")
        else:
            st.error(f"Missing trained model file: {missing.relative_to(PROJECT_DIR)}")
            st.markdown("Train both models from the project folder, then rerun the dashboard.")
            st.code(TRAIN_COMMANDS, language="bash")
        st.stop()
    except (OSError, EOFError, ValueError, KeyError):
        st.error("The dataset or trained model files could not be read. Check the data and retrain both models.")
        st.stop()

    user_ids = svd.user_ids.intersection(pmf.user_ids).sort_values().tolist()
    if not user_ids:
        st.error("The saved models have no users in common. Train both with the same MovieLens data.")
        st.stop()

    user_control, count_control = st.columns([3, 1], gap="large")
    user_input = user_control.text_input(
        "User ID", value=str(user_ids[0]), help="Enter a MovieLens user ID."
    )
    top_n = count_control.number_input("Movies per list", min_value=1, max_value=25, value=10, step=1)

    try:
        user_id = int(user_input)
    except ValueError:
        st.error("Enter a whole-number MovieLens user ID.")
        st.stop()

    if user_id not in user_ids:
        st.error(f"User ID {user_id} was not found. Try {user_ids[0]}.")
        st.stop()

    user_ratings = ratings.loc[ratings["UserID"].eq(user_id)]
    tables = {
        "SVD": generate_recommendations(user_id, svd, top_n),
        "PMF": generate_recommendations(user_id, pmf, top_n),
    }
    shared_picks = set(tables["SVD"]["MovieID"]) & set(tables["PMF"]["MovieID"])
    rating_count, average, shared = st.columns(3)
    rating_count.metric("Movies rated", len(user_ratings), border=True)
    mean_rating = f"{user_ratings['Rating'].mean():.2f} / 5" if not user_ratings.empty else "—"
    average.metric("Average rating", mean_rating, border=True)
    shared.metric("Shared recommendations", len(shared_picks), border=True)

    recommendations_tab, comparison_tab, favorites_tab, reports_tab = st.tabs(
        ["Recommendations", "Held-out comparison", "Top-rated movies", "Model reports"]
    )
    with recommendations_tab:
        for column, (model_name, recommendations) in zip(st.columns(2, gap="large"), tables.items()):
            with column:
                st.subheader(model_name)
                if recommendations.empty:
                    st.info("No unrated movies are available for this model.")
                else:
                    show_movie_table(recommendations, "PredictedRating", "Prediction")
        combined = pd.concat(tables.values(), ignore_index=True)
        if not combined.empty:
            st.download_button(
                "Download recommendations",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name=f"user_{user_id}_recommendations.csv",
                mime="text/csv",
            )

    with comparison_tab:
        predictions = get_user_test_predictions(user_id, test, svd, pmf)
        show_comparison(predictions, movies["Title"])

    with favorites_tab:
        if user_ratings.empty:
            st.info("This user has no recorded ratings.")
        else:
            favorites = (
                user_ratings.sort_values(["Rating", "Timestamp"], ascending=False)
                .head(top_n)
                .join(movies[["Title", "Genres"]], on="MovieID")
            )
            show_movie_table(favorites, "Rating", "User rating")

    with reports_tab:
        show_model_reports()


if __name__ == "__main__":
    main()
