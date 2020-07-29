import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()


def trend(time, slope=0):
    return slope*time


baseline = 10
time = np.arange(4*365+1)
series = trend(time, 0.1)
plot_series(time, series)


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4, np.cos(season_time*2*np.pi), 1/np.exp(3*season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time+phase) % period)/period
    return amplitude * seasonal_pattern(season_time)


amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)
plot_series(time, series)

slope = 0.05
series = baseline+trend(time, slope)+seasonality(time, period=365, amplitude=amplitude)
plot_series(time, series)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time))*noise_level


noise_level = 5
noise = white_noise(time, noise_level, seed=42)
plot_series(time, noise)

series += noise
plot_series(time, series)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude


series = autocorrelation(time, 10, seed=42)
plot_series(time[:200], series[:200])

series = autocorrelation(time, 10, seed=42)+trend(time, 2)
plot_series(time[:200], series[:200])

series = autocorrelation(time, 10, seed=42)+seasonality(time, period=50, amplitude=150)+trend(time, 2)
plot_series(time[:200], series[:200])

series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
series2 = autocorrelation(time, 5, seed=42) + seasonality(time, period=50, amplitude=2) + trend(time, -1) + 550
series[200:] = series2[200:]
plot_series(time[:300], series[:300])
