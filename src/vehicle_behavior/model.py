from __future__ import annotations

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    Lambda,
)
from tensorflow.keras.models import Model


def compute_acceleration(speed_tensor: tf.Tensor) -> tf.Tensor:
    zeros = tf.zeros_like(speed_tensor[:, :1, :])
    accel = speed_tensor[:, 1:, :] - speed_tensor[:, :-1, :]
    return tf.concat([zeros, accel], axis=1)


def compute_lateral_movement(pos_tensor: tf.Tensor) -> tf.Tensor:
    displacement = pos_tensor[:, 1:, :] - pos_tensor[:, :-1, :]
    lateral_speed = tf.norm(displacement, axis=-1, keepdims=True)
    zeros = tf.zeros_like(lateral_speed[:, :1, :])
    return tf.concat([zeros, lateral_speed], axis=1)


CUSTOM_OBJECTS = {
    "compute_acceleration": compute_acceleration,
    "compute_lateral_movement": compute_lateral_movement,
}


def build_cnn_lstm_model(
    sequence_length: int = 60,
    num_classes: int = 3,
    learning_rate: float = 0.001,
) -> Model:
    """CNN-LSTM hybrid model from the notebook pipeline."""
    speed_input = Input(shape=(sequence_length, 1), name="speed")
    position_input = Input(shape=(sequence_length, 2), name="position")

    accel = Lambda(compute_acceleration, name="acceleration")(speed_input)
    lateral = Lambda(compute_lateral_movement, name="lateral")(position_input)
    features = Concatenate(axis=-1)([speed_input, accel, lateral])

    cnn = Conv1D(64, 3, activation="relu", padding="same")(features)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.3)(cnn)
    cnn = Conv1D(128, 3, activation="relu", padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn_out = GlobalMaxPooling1D()(cnn)

    lstm = LSTM(128, return_sequences=True, use_cudnn=False)(features)
    lstm = BatchNormalization()(lstm)
    lstm_out = LSTM(64, use_cudnn=False)(lstm)

    combined = Concatenate()([cnn_out, lstm_out])
    x = Dense(256, activation="relu")(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=[speed_input, position_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_model(
    sequence_length: int = 60,
    num_classes: int = 3,
    learning_rate: float = 0.001,
) -> Model:
    """LSTM-only sequence baseline."""
    speed_input = Input(shape=(sequence_length, 1), name="speed")
    position_input = Input(shape=(sequence_length, 2), name="position")
    features = Concatenate(axis=-1)([speed_input, position_input])

    lstm = LSTM(128, return_sequences=True, use_cudnn=False)(features)
    lstm = BatchNormalization()(lstm)
    lstm_out = LSTM(64, use_cudnn=False)(lstm)
    x = Dense(128, activation="relu")(lstm_out)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=[speed_input, position_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_logistic_regression_baseline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def build_random_forest_baseline() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
    )


def is_keras_model(model: object) -> bool:
    return isinstance(model, tf.keras.Model)


def save_artifacts(
    model: object,
    label_encoder: object,
    artifacts_dir: str,
    model_name: str,
) -> str:
    from pathlib import Path

    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)

    if is_keras_model(model):
        model_path = path / f"{model_name}.keras"
        model.save(model_path)
    else:
        model_path = path / f"{model_name}.joblib"
        joblib.dump(model, model_path)

    encoder_path = path / "label_encoder.joblib"
    joblib.dump(label_encoder, encoder_path)
    return str(model_path)


def load_model(model_path: str) -> object:
    if model_path.endswith(".keras"):
        return tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
    return joblib.load(model_path)
