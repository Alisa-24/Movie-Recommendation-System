# MovieLens Recommender

This project compares two matrix factorization models on MovieLens 1M. Use the
Streamlit dashboard to inspect a user's ratings, get recommendations from each
model, and compare their predicted ratings against that user's held-out ratings.

## Quick start

Run the commands below from the repository root. The project uses Python 3.12 or
newer and provides a lockfile for [uv](https://docs.astral.sh/uv/).

1. Install the project dependencies.

   ```bash
   uv sync --locked
   ```

   To use pip, activate a Python 3.12+ virtual environment and run
   `python -m pip install -r requirements.txt`. The commands below can then be
   run without the `uv run` prefix.

2. Extract `ratings.dat`, `movies.dat`, and `users.dat` into the `data/` folder.
   [Download MovieLens 1M from GroupLens](https://grouplens.org/datasets/movielens/1m/)

3. Train SVD and PMF, then export the aggregate comparison plots and a sample
   user's recommendations and charts.

   ```bash
   uv run python -m models.svd_model
   uv run python -m models.pmf_model
   uv run python -m utils.recommendation --user-id 1 --top-n 10
   ```

4. Start the dashboard and open the local URL printed in the terminal.

   ```bash
   uv run streamlit run app.py
   ```

   With the project environment already active, use `streamlit run app.py`.
   You can skip training when the saved model files already exist. Stop the
   dashboard before retraining so it releases the saved models. Re-export the
   reports and restart the dashboard to load the updated files.

## Dashboard

Enter a **User ID**, press Enter, and set **Movies per list**. The dashboard
updates when either input changes. An unknown ID or invalid number shows an
error and leaves the input available for another attempt.

- **Recommendations** lists each model's highest-ranked movies that the user has
  yet to rate. Download the two lists as a single CSV.
- **Held-out comparison** plots actual, SVD, and PMF ratings for up to 10 of the
  user's test movies. The RMSE cards use every test movie that both models can score
  for that user.
- **Top-rated movies** shows the user's highest historical ratings, with recent
  ratings first in case of a tie.
- **Model reports** shows the PMF learning curves, aggregate test RMSE, and
  predicted versus actual ratings for a sample of test pairs known to both
  models. The aggregate RMSE uses the full test set.

## Training and evaluation

The loader removes missing and duplicate records and keeps the latest rating for
each user and movie pair. Both models use the same 80/20 train/test split with
`random_state=42`.

- **SVD** subtracts each user's training mean before filling missing centered
  entries with zero. SciPy's `svds` keeps the 50 largest singular values and
  their vectors. For a known user and movie, the prediction comes from their
  entry in `U @ diag(s) @ Vt` plus the user's training mean, clipped to 1–5.
- **PMF** fits user and movie biases and latent factors with regularized stochastic
  gradient descent on observed ratings. Each epoch shuffles those ratings. For
  each rating, the model takes the global mean, adds both biases and the dot
  product of the user's and movie's factor vectors, then updates their parameters
  using the prediction error. L2 regularization shrinks large factor and bias
  values, limiting how closely the model fits noise in the training data. PMF
  trains only on observed entries and can account for movie-wide rating tendencies
  through its biases.

PMF holds out 10% of the training ratings to select the best epoch by validation
RMSE. The convergence plot shows training and validation MSE. The saved run chose
epoch 23 and stopped epoch selection after 10 more epochs without an improvement.
It then refit for 23 epochs on the full training split before scoring the test
set. Recommendations rank movies by raw PMF score and display ratings clipped to
1–5.

The saved run produced these results on the shared test split. PMF's aggregate
RMSE was about 11.45% lower than SVD's. [Saved metrics](reports/model_metrics.json)

| Model | Test MSE | Test RMSE |
| --- | ---: | ---: |
| SVD | 0.9294 | 0.9640 |
| PMF | 0.7287 | 0.8537 |

## Recommendation examples

- **User 1** rated *Toy Story* and *Back to the Future* 5 stars. Both are already
  in the user's history, so neither can be recommended. SVD ranks both unrated
  *Back to the Future* sequels in its top ten. PMF favors dramas such as
  *The Shawshank Redemption*. Drama appeared in 21 of the user's 45 titles rated
  at least 4 stars. [User 1 recommendations](reports/user_1_recommendations.csv)
- **User 67** rated 50 dramas at least 4 stars and gave classic comedies such as
  *His Girl Friday* 5 stars. SVD's top picks include *Galaxy Quest* and *Pulp
  Fiction*. PMF puts *The General* first and includes three Wallace & Gromit
  titles. Both models recommend *The Wrong Trousers*. [User 67 recommendations](reports/user_67_recommendations.csv)

## Notebook and reports

Open [Movie_Recommender_System.ipynb](Movie_Recommender_System.ipynb) to inspect
the dataset, tune the models, and plot PMF training and validation loss. Select the
project's `.venv` as the Jupyter kernel. If your editor needs a registered kernel,
run the commands below and choose **Python (matrix-factorization)**.

```bash
uv pip install ipykernel
uv run --no-sync python -m ipykernel install --user --name matrix-factorization --display-name "Python (matrix-factorization)"
```

Notebook exports are off by default. Enable `EXPORT_REPORTS` in the last cell to
save the current experiment's models, charts, and recommendation CSV.

To export the evaluation charts, recommendation CSV, and user comparison plots
from trained models, run

```bash
uv run python -m utils.recommendation --user-id 1 --top-n 10
```

## Project files

```text
matrix-factorization/
├── data/                            # MovieLens 1M source data
│   ├── ratings.dat                  # User and movie IDs, ratings, and timestamps
│   ├── users.dat                    # User IDs and demographic fields
│   └── movies.dat                   # Movie IDs, titles, and genres
├── processed/                       # Data prepared for model training
│   └── user_item_matrix.csv         # User-centered training ratings, with zeros for missing entries
├── models/                          # Model training and test evaluation
│   ├── svd_model.py                 # Fits truncated SVD with SciPy and reconstructs ratings
│   └── pmf_model.py                 # Fits regularized PMF and selects an epoch using validation RMSE
├── utils/                           # Data preparation, recommendation, and plotting code
│   ├── data_loader.py               # Loads and cleans the MovieLens files
│   ├── matrix_creation.py           # Creates the reproducible 80/20 split and centers training ratings
│   ├── model_evaluation.py          # Builds the aggregate test comparison plots
│   └── recommendation.py            # Ranks unrated movies and exports recommendations and plots
├── reports/                         # Saved models, test metrics, charts, and recommendation exports
│   ├── pmf_factors/                 # Arrays and metadata used to reload PMF
│   │   ├── U.npy                    # Learned user factor matrix
│   │   ├── V.npy                    # Learned movie factor matrix
│   │   ├── user_bias.npy            # Learned rating offsets for users
│   │   ├── movie_bias.npy           # Learned rating offsets for movies
│   │   ├── user_ids.npy             # User IDs in the order used by the PMF arrays
│   │   ├── movie_ids.npy            # Movie IDs in the order used by the PMF arrays
│   │   └── metadata.json            # PMF training settings, global mean, and selected epoch
│   ├── model_metrics.json           # Test MSE and RMSE for both models and relative RMSE improvement
│   ├── svd_predictions.npy          # Predicted ratings for all users and movies known to SVD
│   ├── svd_user_ids.npy             # Row-to-user mapping for the SVD predictions
│   ├── svd_movie_ids.npy            # Column-to-movie mapping for the SVD predictions
│   ├── pmf_convergence.png          # Training and validation MSE by epoch
│   ├── rmse_comparison.png          # SVD and PMF RMSE on the full test split
│   ├── predicted_vs_actual.png      # Predicted versus actual ratings for sampled test pairs
│   ├── user_comparison.png          # Actual and predicted held-out ratings for the selected user
│   ├── top_recommendations.png      # Predicted ratings for the selected user's top picks
│   ├── user_1_recommendations.csv   # Ranked SVD and PMF picks for user 1
│   └── user_67_recommendations.csv  # Ranked SVD and PMF picks for user 67
├── .gitignore                       # Excludes local data, generated arrays, and environment files
├── app.py                           # Streamlit dashboard for recommendations and model comparisons
├── Movie_Recommender_System.ipynb   # Data exploration, model experiments, and evaluation
├── README.md                        # Setup, model explanations, and project layout
├── requirements.txt                 # Pinned dependencies for pip installs
├── pyproject.toml                   # Project metadata and dependency requirements
└── uv.lock                          # Resolved dependency versions for uv installs
```

The dataset, processed matrix, and NumPy model files are generated or downloaded
locally and are excluded from Git.
