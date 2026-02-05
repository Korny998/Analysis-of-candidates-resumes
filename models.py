from tensorflow.keras import Input, layers, Model


def build_assembly_model(
    tabular_dim: int,
    education_dim: int,
    works_dim: int,
    position_dim: int
):
    """Build a multi-input Keras model for salary regression."""
    input1 = Input(shape=(tabular_dim,), name='tabular_input')
    input2 = Input(shape=(education_dim,), name='education_input')
    input3 = Input(shape=(works_dim,), name='works_input')
    input4 = Input(shape=(position_dim,), name='position_input')

    x1 = layers.Dense(20, activation='relu')(input1)
    x1 = layers.Dense(500, activation='relu')(x1)
    x1 = layers.Dense(200, activation='relu')(x1)

    x2 = layers.Dense(20, activation='relu')(input2)
    x2 = layers.Dense(200, activation='relu')(x2)
    x2 = layers.Dropout(0.3)(x2)

    x3 = layers.Dense(20, activation='relu')(input3)
    x3 = layers.Dense(200, activation='relu')(x3)
    x3 = layers.Dropout(0.3)(x3)

    x4 = layers.Dense(20, activation='relu')(input4)
    x4 = layers.Dense(200, activation='relu')(x4)
    x4 = layers.Dropout(0.3)(x4)

    x = layers.Concatenate()([x1, x2, x3, x4])
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(1, activation='linear', name='salary_scaled')(x)

    model = Model(
        inputs=[input1, input2, input3, input4],
        outputs=output,
        name='salary_regression_model'
    )
    return model


def build_simplified_assembly_model(
    tabular_dim: int,
    position_dim: int
):
    """Build a simplified Keras model with multiple inputs for salary regression."""
    input1 = Input(shape=(tabular_dim,), name='tabular_input')
    input2 = Input(shape=(position_dim,), name='position_input')

    x1 = layers.Dense(20, activation='relu')(input1)
    x1 = layers.Dense(500, activation='relu')(x1)
    x1 = layers.Dense(200, activation='relu')(x1)

    x2 = layers.Dense(20, activation='relu')(input2)
    x2 = layers.Dense(200, activation='relu')(x2)
    x2 = layers.Dropout(0.3)(x2)

    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(1, activation='linear', name='salary_scaled')(x)

    model = Model(
        inputs=[input1, input2],
        outputs=x,
        name='simplified_salary_regression_model'
    )
    
    return model
