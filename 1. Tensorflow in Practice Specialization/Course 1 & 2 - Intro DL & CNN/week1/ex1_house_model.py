import tensorflow as tf
import numpy as np
from tensorflow import keras

"""
Make a function which trains a neural network to predict
house prices following the rule:
price = $50k + $50k * number of bedrooms
"""


def model_house(x_data, y_data):
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd',
                  loss='mean_squared_error')
    model.fit(x_data, y_data, epochs=1000)  # 100, 500, 1000
    return model


xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 5.5], dtype=float)
model_train = model_house(xs, ys)
print(model_train.predict([7.0])[0])
print(model_train.predict([50.0])[0])
