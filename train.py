from tensorflow.keras.optimizers import MSE
from tensorflow.keras.optimizers import Adam

from dataset import x_train, x_train_education, x_train_works, x_train_position


def train_model():
    model = build_model(
        tabular_dim=x_train.shape[1],
        education_dim=x_train_education.shape[1],
        works_dim=x_train_works.shape[1],
        position_dim=x_train_position.shape[1],
    )
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='mse',
        metrics=['mae']
    )
    
    history = model.fit([
        [
            x_train[:8000],
            x_train_education[:8000],
            x_train_works[:8000],
            x_train_position[:8000]
        ],
        y_train_scaled[:8000],
        batch_size=256,
        epochs=100,
        validation_data=([
            x_train[8000:],
            x_train_education[8000:],
            x_train_works[8000:],
            x_train_position[8000:]
        ], y_train_scaled[8000:]),
        verbose=1
    ])
    return model, history


if __name__ == '__main__':
    model_functional, history = train_model()
    pred = model_functional.predict([
        x_train[8000:8100],
        x_train_education[8000:8100],
        x_train_works[8000:8100],
        x_train_position[8000:8100]
    ])
    y_pred = y_scaler.inverse_transform(pred)
