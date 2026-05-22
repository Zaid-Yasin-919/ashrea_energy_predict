import math

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

from load_data import load_datasets


SELECTED_FEATURES = [
    "site_id",
    "relative_humidity",
    "feels_like_capped",
    "wind_speed",
    "precip_1h_category",
    "is_light_rain",
    "is_moderate_rain",
    "is_heavy_rain",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "season",
    "timestamp",
]


def saturation_vapor_pressure(temperature):
    return 6.112 * math.exp((17.67 * temperature) / (temperature + 243.5))


def relative_humidity(air_temp, dew_temp):
    e_air = saturation_vapor_pressure(air_temp)
    e_dew = saturation_vapor_pressure(dew_temp)
    return (e_dew / e_air) * 100


def add_relative_humidity(weather_df):
    weather_df["relative_humidity"] = weather_df.apply(
        lambda row: relative_humidity(row["air_temperature"], row["dew_temperature"]),
        axis=1,
    )
    return weather_df


def add_feels_like(weather_df):
    weather_df["heat_index"] = (
        0.5
        * (
            weather_df["air_temperature"]
            + 61.0
            + (weather_df["air_temperature"] - 68.0) * 1.2
            + weather_df["dew_temperature"] * 0.094
        )
    )

    mask = (weather_df["air_temperature"] < 10) & (weather_df["wind_speed"] > 1.34)
    weather_df["wind_chill"] = (
        13.12
        + 0.6215 * weather_df["air_temperature"]
        - 11.37 * (weather_df["wind_speed"] ** 0.16)
        + 0.3965
        * weather_df["air_temperature"]
        * (weather_df["wind_speed"] ** 0.16)
    )
    weather_df["wind_chill"] = weather_df["wind_chill"].where(
        mask, weather_df["air_temperature"]
    )

    weather_df["feels_like"] = np.where(
        weather_df["air_temperature"] >= 27,
        weather_df["heat_index"],
        np.where(
            weather_df["air_temperature"] <= 10,
            weather_df["wind_chill"],
            weather_df["air_temperature"],
        ),
    )

    weather_df["feels_like_capped"] = weather_df["feels_like"].clip(-10, 35)
    return weather_df


def add_time_features(weather_df):
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    weather_df["hour"] = weather_df["timestamp"].dt.hour
    weather_df["day_of_week"] = weather_df["timestamp"].dt.weekday
    weather_df["month"] = weather_df["timestamp"].dt.month
    weather_df["day_of_year"] = weather_df["timestamp"].dt.dayofyear
    weather_df["is_weekend"] = (weather_df["day_of_week"] >= 5).astype(int)

    weather_df["hour_sin"] = np.sin(2 * np.pi * weather_df["hour"] / 24)
    weather_df["hour_cos"] = np.cos(2 * np.pi * weather_df["hour"] / 24)
    weather_df["day_sin"] = np.sin(2 * np.pi * weather_df["day_of_week"] / 7)
    weather_df["day_cos"] = np.cos(2 * np.pi * weather_df["day_of_week"] / 7)
    weather_df["month_sin"] = np.sin(2 * np.pi * weather_df["month"] / 12)
    weather_df["month_cos"] = np.cos(2 * np.pi * weather_df["month"] / 12)

    weather_df["season"] = weather_df["month"].map(
        lambda month: 0
        if month in [12, 1, 2]
        else 1
        if month in [3, 4, 5]
        else 2
        if month in [6, 7, 8]
        else 3
    )
    return weather_df


def add_precipitation_features(weather_df):
    weather_df["precip_depth_1_hr"] = (
        weather_df["precip_depth_1_hr"]
        .replace(-1.0, np.nan)
        .interpolate(method="linear", limit=6)
        .fillna(0)
    )

    bins = [-0.1, 0.1, 5.0, 15.0, float("inf")]
    labels = ["No Rain", "Light Rain", "Moderate Rain", "Heavy Rain"]

    weather_df["precip_1h_category"] = pd.cut(
        weather_df["precip_depth_1_hr"],
        bins=bins,
        labels=labels,
    )

    weather_df["no_rain"] = (weather_df["precip_depth_1_hr"] == 0).astype(int)
    weather_df["is_light_rain"] = (weather_df["precip_depth_1_hr"] > 0.1).astype(int)
    weather_df["is_moderate_rain"] = (
        (weather_df["precip_depth_1_hr"] > 5.0)
        & (weather_df["precip_depth_1_hr"] < 15.0)
    ).astype(int)
    weather_df["is_heavy_rain"] = (weather_df["precip_depth_1_hr"] >= 15.0).astype(int)
    return weather_df


def clean_weather_values(weather_df):
    weather_df["sea_level_pressure"] = winsorize(
        weather_df["sea_level_pressure"], limits=[0.01, 0.01]
    )
    weather_df[["feels_like_capped", "relative_humidity", "wind_speed"]] = weather_df[
        ["feels_like_capped", "relative_humidity", "wind_speed"]
    ].interpolate(method="linear")
    return weather_df


def prepare_weather_data(weather_df):
    weather_df = weather_df.copy()
    weather_df = add_relative_humidity(weather_df)
    weather_df = add_feels_like(weather_df)
    weather_df = add_time_features(weather_df)
    weather_df = add_precipitation_features(weather_df)
    weather_df = clean_weather_values(weather_df)
    return weather_df


def build_weather_datasets(weather_train_df, weather_test_df):
    weat_cop = prepare_weather_data(weather_train_df)
    test_weat_cop = prepare_weather_data(weather_test_df)

    final_weat_train = weat_cop[SELECTED_FEATURES]
    final_weat_test = test_weat_cop[SELECTED_FEATURES]

    return weat_cop, test_weat_cop, final_weat_train, final_weat_test


def main():
    _, _, _, weather_train_df, weather_test_df = load_datasets()
    weat_cop, test_weat_cop, final_weat_train, final_weat_test = build_weather_datasets(
        weather_train_df,
        weather_test_df,
    )

    print("weat_cop", weat_cop.shape)
    print("test_weat_cop", test_weat_cop.shape)
    print("Final_weat_train", final_weat_train.shape)
    print("Final_weat_test", final_weat_test.shape)


if __name__ == "__main__":
    main()
