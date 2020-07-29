import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Make a function which trains a neural network to predict
house prices following the rule:
price = $50k + $50k * number of bedrooms
"""

xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 5.5], dtype=float)

lr = 6e-2
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])
model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.Adam(lr=lr),
    metrics=['mae', 'mse']
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: lr * 10 ** (epoch/100)
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=20,
    mode='auto'
)

history = model.fit(xs, ys, epochs=500, callbacks=[early_stop])

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([lr, 1e+2, 0, 1])
# plt.show()


def plot_graphs(hist, string, start=None):
    label_plot = hist.history[string]
    epochs_plot = range(len(label_plot))
    if start is not None:
        label_plot = label_plot[start:]
        epochs_plot = epochs_plot[start:]

    plt.plot(epochs_plot, label_plot)
    plt.xlabel("Epochs")
    plt.ylabel(string)


plot_graphs(history, 'mae', 50)
plot_graphs(history, 'loss', 50)
plt.show()

print(model.predict([7.0])[0])
print(model.predict([50.0])[0])
