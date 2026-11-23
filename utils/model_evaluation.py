"""Plots for the aggregate held-out evaluation."""

import matplotlib.pyplot as plt


def plot_rmse_comparison(metrics):
    """Compare test RMSE across models on the full shared test split."""
    models = ["SVD", "PMF"]
    rmse = [metrics[f"{model}_RMSE"] for model in models]
    figure, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    bars = ax.bar(models, rmse, color=["tab:blue", "tab:orange"])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(
        title="Model RMSE on the MovieLens test set",
        xlabel="Model",
        ylabel="Test RMSE",
        ylim=(0, max(rmse) * 1.18 or 1),
    )
    return figure


def plot_predicted_vs_actual(predictions):
    """Plot paired model predictions for a sample of held-out ratings."""
    figure, axes = plt.subplots(
        1, 2, figsize=(9, 4), sharex=True, sharey=True, layout="constrained"
    )
    for ax, model in zip(axes, ("SVD", "PMF")):
        ax.scatter(predictions["Rating"], predictions[model], s=9, alpha=0.15)
        ax.plot([1, 5], [1, 5], color="gray", linestyle="--", linewidth=1)
        ax.set(title=model, xlabel="Actual rating", xticks=range(1, 6), yticks=range(1, 6))

    axes[0].set_ylabel("Predicted rating")
    figure.suptitle(f"Predicted vs. actual ratings on {len(predictions):,} test rows")
    return figure
