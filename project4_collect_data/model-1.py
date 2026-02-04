import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, GlobalAveragePooling1D

"""
Build a 1D CNN model for posture classification.

Args:
    hidden_activation (str): Activation function used in hidden layers
        (e.g., 'relu').
    input_shape (tuple): Shape of the input window, given as
        (window_size, num_channels).
    num_classes (int): Number of output classes (e.g., 5 postures).

Returns:
    tensorflow.keras.Model: A compiled Keras model ready for training.
"""
def build_model(hidden_activation, input_shape, num_classes):
    # Input: time window of 3-axis sensor data
    inputs = Input(shape=input_shape)

    x = Conv1D(16, 5, activation=hidden_activation, padding='same')(inputs)
    x = Conv1D(32, 5, activation=hidden_activation, padding='same')(x)
    # Aggregate temporal dimension to a single feature vector
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    # Fully connected layer before the classifier
    x = Dense(32, activation=hidden_activation)(x)
    # Output layer with softmax for multi-class classification
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compile with standard settings for classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
