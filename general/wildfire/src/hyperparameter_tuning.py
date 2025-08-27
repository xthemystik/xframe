from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_dim: int):
    """
    Build and compile a simple feedforward binary classifier.

    This replaces the previous kerastuner-dependent version so core training
    can proceed without external tuning packages.
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model