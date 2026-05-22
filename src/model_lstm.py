import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

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


def split_lstm_features(final_train):
    X = final_train[FEATURES]
    y = final_train[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    return X_train, X_val, y_train, y_val, scaler


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, activation="tanh", input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def train_lstm_model(X_train, X_val, y_train, y_val, epochs=10, batch_size=32):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )

    return model, history


def evaluate_model(model, X_train, X_val, y_train, y_val):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    r2_train = r2_score(y_train, np.ravel(y_train_pred))
    r2_val = r2_score(y_val, np.ravel(y_val_pred))

    return r2_train, r2_val


def main():
    train_df, _, building_df, weather_train_df, weather_test_df = load_datasets()
    final_train, _ = build_final_train(
        train_df,
        building_df,
        weather_train_df,
        weather_test_df,
    )

    X_train, X_val, y_train, y_val, _ = split_lstm_features(final_train)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)

    model, _ = train_lstm_model(X_train, X_val, y_train, y_val)
    model.summary()

    r2_train, r2_val = evaluate_model(model, X_train, X_val, y_train, y_val)
    print("Training R2 Score:", r2_train)
    print("Test R2 Score:", r2_val)


if __name__ == "__main__":
    main()
