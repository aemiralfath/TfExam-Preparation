import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,  # 0.4
                    np.cos(season_time * 6 * np.pi),  # 2
                    2 / np.exp(9 * season_time))  # 1, 3


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(10*365+1, dtype="float32")  # 4
baseline = 10
amplitude = 40
slope = 0.005  # 0.05, 0.09
noise_level = 3  # 5

series = baseline+trend(time, slope)+seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=51)  # 42

split_time = 3000  # 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32  # 128
shuffle_buffer_size = 1000

plot_series(time, series)
plt.show()


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    return dataset.batch(batch_size).prefetch(1)


tf.keras.backend.clear_session()
tf.random.set_seed(51)  # 42
np.random.seed(51)  # 42

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),  # tf.keras.layers.SimpleRNN(40, return_sequences=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),  # tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8*10**(epoch/20)
)
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(
    loss="mse",  # tf.keras.losses.Huber()
    optimizer=optimizer,
    metrics=["mae"]
)

history = model.fit(dataset, epochs=400)  # 300, callbacks=[lr_schedule]

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 30])
# plt.show()

forecast = []
for time in range(len(series)-window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

print(tf.keras.metrics.mean_squared_error(x_valid, results).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())


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


plot_graphs(history, 'mae', 50)
plot_graphs(history, 'loss', 50)
