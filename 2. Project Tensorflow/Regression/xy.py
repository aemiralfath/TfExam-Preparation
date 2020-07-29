import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

lr = 6e-2
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
    patience=30,
    mode='auto'
)

history = model.fit(xs, ys, epochs=500, callbacks=[early_stop])

# plt.semilogx(history.history["lr"], history.history["loss"])
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


plot_graphs(history, 'mae', 20)
plot_graphs(history, 'loss', 20)
plt.show()

print(model.predict([5.0]))
print(model.predict([6.0]))
print(model.predict([7.0]))
print(model.predict([8.0]))
print(model.predict([9.0]))
print(model.predict([10.0]))
