from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_datasets(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, ...]:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    building_df = pd.read_csv(data_dir / "building_metadata.csv")
    weather_train_df = pd.read_csv(data_dir / "weather_train.csv")
    weather_test_df = pd.read_csv(data_dir / "weather_test.csv")

    return train_df, test_df, building_df, weather_train_df, weather_test_df


def main() -> None:
    train_df, test_df, building_df, weather_train_df, weather_test_df = load_datasets()

    weat_cop = weather_train_df.copy()
    test_weat_cop = weather_test_df.copy()

    print("train_df", train_df.shape)
    print("test_df", test_df.shape)
    print("building_df", building_df.shape)
    print("weather_train_df", weather_train_df.shape)
    print("weather_test_df", weather_test_df.shape)
    print("weat_cop", weat_cop.shape)
    print("test_weat_cop", test_weat_cop.shape)


if __name__ == "__main__":
    main()
