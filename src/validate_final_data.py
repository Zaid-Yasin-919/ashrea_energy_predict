from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from final_train_data import build_final_train
from load_data import load_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def print_final_data_summary(final_train):
    print("Final_train", final_train.shape)
    print("\nDtypes")
    print(final_train.dtypes)
    print("\nMissing values")
    print(final_train.isnull().sum())
    print("\nMeter reading summary")
    print(final_train["meter_reading"].describe())
    print("\nMeter type counts")
    print(final_train["meter"].value_counts().sort_index())


def save_correlation_heatmap(final_train, figures_dir=FIGURES_DIR):
    numeric_cols = final_train.select_dtypes(include=["float64", "int64", "int32"]).columns
    corr_df = final_train[numeric_cols].corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        mask=np.triu(np.ones_like(corr_df)),
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "final_train_correlation_heatmap.png", dpi=150)
    plt.close()


def save_monthly_meter_reading(final_train, figures_dir=FIGURES_DIR):
    monthly_reading = final_train.groupby("month")["meter_reading"].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x=monthly_reading.index,
        y=monthly_reading.values,
        marker="o",
        linewidth=2.5,
        color="blue",
    )
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Meter Reading", fontsize=12)
    plt.title("Monthly Average Meter Reading", fontsize=14)
    plt.xticks(
        ticks=range(1, 13),
        labels=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_train_monthly_meter_reading.png", dpi=150)
    plt.close()


def save_hourly_meter_reading(final_train, figures_dir=FIGURES_DIR):
    hourly_reading = final_train.groupby("hour")["meter_reading"].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x=hourly_reading.index,
        y=hourly_reading.values,
        marker="o",
        linewidth=2.5,
        color="green",
    )
    plt.xlabel("Hour", fontsize=12)
    plt.ylabel("Average Meter Reading", fontsize=12)
    plt.title("Hourly Energy Usage Pattern", fontsize=14)
    plt.xticks(ticks=range(0, 24))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_train_hourly_meter_reading.png", dpi=150)
    plt.close()


def save_meter_type_distribution(final_train, figures_dir=FIGURES_DIR):
    meter_counts = final_train["meter"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=meter_counts.index, y=meter_counts.values)
    plt.xlabel("Meter Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Meter Types", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_train_meter_type_distribution.png", dpi=150)
    plt.close()


def save_meter_reading_distribution(final_train, figures_dir=FIGURES_DIR):
    plt.figure(figsize=(10, 5))
    sns.histplot(final_train["meter_reading"], bins=80, kde=True)
    plt.xlabel("Meter Reading", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Meter Reading Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_train_meter_reading_distribution.png", dpi=150)
    plt.close()


def save_validation_plots(final_train, figures_dir=FIGURES_DIR):
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_correlation_heatmap(final_train, figures_dir)
    save_monthly_meter_reading(final_train, figures_dir)
    save_hourly_meter_reading(final_train, figures_dir)
    save_meter_type_distribution(final_train, figures_dir)
    save_meter_reading_distribution(final_train, figures_dir)


def main():
    train_df, _, building_df, weather_train_df, weather_test_df = load_datasets()
    final_train, _ = build_final_train(
        train_df,
        building_df,
        weather_train_df,
        weather_test_df,
    )

    print_final_data_summary(final_train)
    save_validation_plots(final_train)
    print(f"\nSaved validation plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
