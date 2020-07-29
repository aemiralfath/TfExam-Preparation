import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celcius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

lo = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential(lo)

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1),
              metrics=['mae', 'mse'])

history = model.fit(celcius, fahrenheit, epochs=500)

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.show()

print(model.predict([100.0]))
print("These are the layer variable: {}".format(lo.get_weights()))
