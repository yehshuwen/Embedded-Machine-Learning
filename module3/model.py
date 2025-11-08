import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout

"""
Build a model

Args:
    hidden_activation (str): activation function, default: relu
    num_features (int): features of input data, default: 6 (IMU data)
    num_classes (int): classes number of output, default: 5 (5 posture)

Return:
    keras.Model
"""
def build_model(hidden_activation='relu', num_features=6, num_classes=5):
    model = keras.Sequential([
        layers.InputLayer(shape=(num_features,)),
        layers.Dense(64, activation=hidden_activation), # fully connected, has 64 nodes
        #Dropout(0.2)
        layers.Dense(32, activation=hidden_activation),
        layers.Dense(16, activation=hidden_activation),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model