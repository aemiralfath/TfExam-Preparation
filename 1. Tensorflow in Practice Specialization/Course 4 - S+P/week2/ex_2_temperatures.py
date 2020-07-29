import csv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


time_step = []
temps = []

with open('../../dataset/daily-min-temperatures.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    step = 0
    for row in reader:
        temps.append(float(row[1]))
        time_step.append(step)
        step += 1

series = np.array(temps)
time = np.array(time_step)
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

split_time = 2500
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    return model.predict(ds)


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

window_size = 60
batch_size = 100
shuffle_buffer_size = 1000

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5, strides=1, padding="causal",
                           activation="relu", input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*400)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6*10**(epoch/20)
)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=3,
    mode='auto'
)
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=optimizer,
    metrics=["mae"]
)
history = model.fit(train_set, epochs=150, callbacks=[earlystop])

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-2, 0, 60])
# plt.show()

forecast = model_forecast(model, series[..., np.newaxis], window_size)
forecast = forecast[split_time-window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, forecast)
plt.show()

print(tf.keras.metrics.mean_absolute_error(x_valid, forecast).numpy())
print(tf.keras.metrics.mean_squared_error(x_valid, forecast).numpy())
print(forecast)


def plot_graphs(history, string, start=None):
    label_plot = history.history[string]
    epochs_plot = range(len(label_plot))
    if start is not None:
        label_plot = label_plot[start:]
        epochs_plot = epochs_plot[start:]

    plt.plot(epochs_plot, label_plot)
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


plot_graphs(history, 'mae', 0)
plot_graphs(history, 'loss', 0)
