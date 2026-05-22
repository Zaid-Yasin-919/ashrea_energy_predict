import numpy as np
from sklearn.preprocessing import LabelEncoder

from load_data import load_datasets


def prepare_building_data(building_df):
    buil_cop = building_df.copy()

    buil_cop.drop(columns=["floor_count", "year_built"], inplace=True)

    label_encoder = LabelEncoder()
    buil_cop["type_enc"] = label_encoder.fit_transform(buil_cop["primary_use"])

    buil_cop["square_feet_log"] = np.log1p(buil_cop["square_feet"])

    buil_cop.drop(columns=["primary_use", "square_feet"], inplace=True)

    return buil_cop, label_encoder


def main():
    _, _, building_df, _, _ = load_datasets()
    buil_cop, _ = prepare_building_data(building_df)

    print("buil_cop", buil_cop.shape)
    print(buil_cop.head())
    print(buil_cop.isnull().sum())


if __name__ == "__main__":
    main()
