import lightgbm as lgb
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from final_train_data import build_final_train
from load_data import load_datasets


FEATURES = [
    "building_id",
    "meter",
    "site_id",
    "type_enc",
    "square_feet_log",
    "relative_humidity",
    "feels_like_capped",
    "wind_speed",
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
]
TARGET = "meter_reading"

PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 51,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 1300,
    "verbose": -1,
}


def split_features(final_train):
    X = final_train[FEATURES]
    y = final_train[TARGET]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def rmsle_lgbm(y_pred, data):
    y_true = np.array(data.get_label())
    score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

    return "rmsle", score, False


def train_lgbm_model(X_train, X_val, y_train, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        PARAMS,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100),
        ],
        feval=rmsle_lgbm,
    )

    return model


def evaluate_model(model, X_train, X_val, y_train, y_val):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)

    return r2_train, r2_val


def main():
    train_df, _, building_df, weather_train_df, weather_test_df = load_datasets()
    final_train, _ = build_final_train(
        train_df,
        building_df,
        weather_train_df,
        weather_test_df,
    )

    X_train, X_val, y_train, y_val = split_features(final_train)
    model = train_lgbm_model(X_train, X_val, y_train, y_val)
    r2_train, r2_val = evaluate_model(model, X_train, X_val, y_train, y_val)

    print("Training R2 Score:", r2_train)
    print("Test R2 Score:", r2_val)


if __name__ == "__main__":
    main()
