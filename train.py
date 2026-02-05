from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from constants import BATCH_SIZE, EPOCHS
from dataset import (
    x_train,
    x_train_education,
    x_train_works,
    x_train_position,
    y_scaler,
    y_train_scaled
)
from models import (
    build_assembly_model,
    build_simplified_assembly_model
)


def train_model():
    model = build_assembly_model(
        tabular_dim=x_train.shape[1],
        education_dim=x_train_education.shape[1],
        works_dim=x_train_works.shape[1],
        position_dim=x_train_position.shape[1]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=MeanSquaredError(),
        metrics=['mae']
    )
 
    history = model.fit(
        x=[
            x_train[:8000],
            x_train_education[:8000],
            x_train_works[:8000],
            x_train_position[:8000]
        ],
        y=y_train_scaled[:8000],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([
            x_train[8000:],
            x_train_education[8000:],
            x_train_works[8000:],
            x_train_position[8000:]
        ], y_train_scaled[8000:]),
        verbose=1
    )
    return model, history


def train_simplified_model():
    model = build_simplified_assembly_model(
        tabular_dim=x_train.shape[1],
        education_dim=x_train_education.shape[1],
        works_dim=x_train_works.shape[1],
        position_dim=x_train_position.shape[1]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=MeanSquaredError(),
        metrics=['mae']
    )

    history = model.fit(
        x=[
            x_train[:8000],
            x_train_position[:8000]
        ],
        y=y_train_scaled[:8000],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([
            x_train[8000:],
            x_train_position[8000:]
        ], y_train_scaled[8000:]),
        verbose=1
    )
    return model, history


if __name__ == '__main__':
    assembly_model, history = train_model()
    pred = assembly_model.predict([
        x_train[8000:8100],
        x_train_education[8000:8100],
        x_train_works[8000:8100],
        x_train_position[8000:8100]
    ])
    y_pred = y_scaler.inverse_transform(pred)

    simplified_model, history = train_simplified_model()
    pred_simplified = simplified_model.predict([
        x_train[8000:8100],
        x_train_position[8000:8100]
    ])
    y_pred_simplified = y_scaler.inverse_transform(pred_simplified)
