import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

xs = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
ys = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)

lr = 1e-1
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])
model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.Adam(lr=lr),
    metrics=['mae', 'mse']
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: lr * 10 ** (epoch/50)
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=10,
    mode='auto'
)

history = model.fit(xs, ys, epochs=500, callbacks=[early_stop])

# plt.semilogx(history.history["lr"], history.history["mae"])
# plt.axis([lr, 1e+2, 0, 60])
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


plot_graphs(history, 'mae', 10)
plot_graphs(history, 'loss', 10)
plt.show()

print(model.predict([100.0]))
