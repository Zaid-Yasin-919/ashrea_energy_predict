import argparse

from final_train_data import build_final_train
from load_data import load_datasets
from model_lgbm import evaluate_model as evaluate_lgbm_model
from model_lgbm import split_features, train_lgbm_model
from model_lstm import evaluate_model as evaluate_lstm_model
from model_lstm import split_lstm_features, train_lstm_model
from validate_final_data import print_final_data_summary, save_validation_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ASHRAE energy pipeline.")
    parser.add_argument(
        "--model",
        choices=["lgbm", "lstm", "both"],
        default="lgbm",
        help="Model stage to run after validation.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip final dataset summary and validation plots.",
    )
    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=10,
        help="Number of epochs for the LSTM model.",
    )
    parser.add_argument(
        "--lstm-batch-size",
        type=int,
        default=32,
        help="Batch size for the LSTM model.",
    )

    return parser.parse_args()


def run_lgbm(final_train):
    print("\nRunning LightGBM model")
    X_train, X_val, y_train, y_val = split_features(final_train)
    model = train_lgbm_model(X_train, X_val, y_train, y_val)
    r2_train, r2_val = evaluate_lgbm_model(model, X_train, X_val, y_train, y_val)

    print("Training R2 Score:", r2_train)
    print("Test R2 Score:", r2_val)

    return model


def run_lstm(final_train, epochs, batch_size):
    print("\nRunning LSTM model")
    X_train, X_val, y_train, y_val, scaler = split_lstm_features(final_train)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)

    model, history = train_lstm_model(
        X_train,
        X_val,
        y_train,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
    )
    model.summary()

    r2_train, r2_val = evaluate_lstm_model(model, X_train, X_val, y_train, y_val)

    print("Training R2 Score:", r2_train)
    print("Test R2 Score:", r2_val)

    return model, history, scaler


def main():
    args = parse_args()

    print("Loading raw datasets")
    train_df, _, building_df, weather_train_df, weather_test_df = load_datasets()

    print("Building final training dataset")
    final_train, _ = build_final_train(
        train_df,
        building_df,
        weather_train_df,
        weather_test_df,
    )

    if not args.skip_validation:
        print("\nValidating final training dataset")
        print_final_data_summary(final_train)
        save_validation_plots(final_train)

    if args.model in ["lgbm", "both"]:
        run_lgbm(final_train)

    if args.model in ["lstm", "both"]:
        run_lstm(final_train, args.lstm_epochs, args.lstm_batch_size)

    print("\nPipeline complete")


if __name__ == "__main__":
    main()
