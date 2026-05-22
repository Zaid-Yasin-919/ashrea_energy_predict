import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from building_data import prepare_building_data
from load_data import load_datasets
from weat_data import build_weather_datasets


def build_final_train(train_df, building_df, weather_train_df, weather_test_df):
    buil_cop, _ = prepare_building_data(building_df)
    _, _, final_weat_train, _ = build_weather_datasets(weather_train_df, weather_test_df)

    train_merged = train_df.merge(buil_cop, on="building_id", how="left")
    train_merged["timestamp"] = pd.to_datetime(train_merged["timestamp"])

    final_train = train_merged.merge(
        final_weat_train,
        on=["site_id", "timestamp"],
        how="inner",
    )

    final_train["meter_reading"] = np.log1p(final_train["meter_reading"])
    final_train.loc[
        (final_train["site_id"] == 0) & (final_train["meter"] == 0),
        "meter_reading",
    ] *= 0.293071

    label_encoder = LabelEncoder()
    final_train["precip_1h_category_enc"] = label_encoder.fit_transform(
        final_train["precip_1h_category"]
    )
    final_train.drop(columns="precip_1h_category", inplace=True)

    drop_0readings = list(final_train[final_train["meter_reading"] == 0.0].index)
    final_train.drop(drop_0readings, axis=0, inplace=True)

    return final_train, label_encoder


def main():
    train_df, _, building_df, weather_train_df, weather_test_df = load_datasets()
    final_train, _ = build_final_train(
        train_df,
        building_df,
        weather_train_df,
        weather_test_df,
    )

    print("Final_train", final_train.shape)
    print(final_train.isnull().sum())


if __name__ == "__main__":
    main()
