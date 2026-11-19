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

2. Extract `ratings.dat`, `movies.dat`, and `users.dat` into the `data/` folder.
   [Download MovieLens 1M from GroupLens](https://grouplens.org/datasets/movielens/1m/)

3. Train SVD and then PMF to generate the model files in `reports/`.

   ```bash
   uv run python -m models.svd_model
   uv run python -m models.pmf_model
   ```

4. Start the dashboard and open the local URL printed in the terminal.

   ```bash
   uv run streamlit run app.py
   ```

   With the project environment already active, use `streamlit run app.py`.
   You can skip training when the saved model files already exist. Stop the dashboard
   before retraining, then restart it to load the updated files.

## Dashboard

Enter a **User ID** in the searchable selector and set **Movies per list**. The
dashboard updates as you change either input.

- **Recommendations** lists each model's highest-ranked movies that the user has
  yet to rate. Download the two lists as a single CSV.
- **Held-out comparison** plots actual, SVD, and PMF ratings for up to 10 of the
  user's test movies. The RMSE cards use every test movie that both models can score
  for that user.
- **Top-rated movies** shows the user's highest historical ratings, with recent
  ratings first in case of a tie.

## Training and evaluation

The loader removes missing and duplicate records and keeps the latest rating for
each user and movie pair. Both models use the same 80/20 train/test split with
`random_state=42`.

- **SVD** subtracts each user's training mean before filling missing centered
  entries with zero. It applies truncated SVD with SciPy, then restores user means
  to reconstruct ratings.
- **PMF** fits user and movie biases and latent factors with regularized stochastic
  gradient descent on observed ratings. It holds out 10% of the training ratings
  to select the best epoch by validation RMSE and then refits on all training
  ratings for that many epochs. The recommender ranks movies using raw PMF scores
  and clips the displayed ratings to 1–5.

The saved run produced these results on the shared test split. PMF's aggregate
RMSE was about 11.45% lower than SVD's. [Saved metrics](reports/model_metrics.json)

| Model | Test MSE | Test RMSE |
| --- | ---: | ---: |
| SVD | 0.9294 | 0.9640 |
| PMF | 0.7287 | 0.8537 |

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

To generate the recommendation CSV and user comparison plots from trained models
without opening the notebook, run

```bash
uv run python -m utils.recommendation --user-id 1 --top-n 10
```

## Project files

| Path | Purpose |
| --- | --- |
| [app.py](app.py) | Interactive Streamlit dashboard |
| [models/](models/) | SVD and PMF training, evaluation, and artifact saving |
| [utils/](utils/) | MovieLens loading, matrix preparation, and recommendation logic |
| `data/` | MovieLens input files |
| `processed/user_item_matrix.csv` | Rounded, normalized training matrix |
| [reports/](reports/) | Model files, saved metrics, convergence plot, and exported comparisons |

The dataset, processed matrix, and NumPy model files are generated or downloaded
locally and are excluded from Git.
