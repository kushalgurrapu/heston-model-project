from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.losses import MeanSquaredError


def build_paper_model(input_dim=22, output_dim=5):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(1000, activation='relu', use_bias=True,
              bias_initializer='random_uniform'),
        Dense(output_dim, activation='sigmoid', use_bias=True,
              bias_initializer='random_uniform'),
    ])
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adadelta(learning_rate=1.0, rho=0.95),
    )
    return model


def build_modern_model(input_dim=22, output_dim=5):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(output_dim, activation='sigmoid'),
    ])
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=1e-3),
    )
    return model


BUILDERS = {
    'paper': build_paper_model,
    'modern': build_modern_model,
}
