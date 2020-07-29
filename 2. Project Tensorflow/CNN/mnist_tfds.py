import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
print(tf.__version__)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.998:
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = MyCallback()
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
x_train, x_test = dataset['train'], dataset['test']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print('Number of training examples: {}'.format(num_train_examples))
print('Number of test examples:     {}'.format(num_test_examples))


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


x_train = x_train.map(normalize)
x_test = x_test.map(normalize)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

lr = 1e-3
epochs = 10
batch_size = 32
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=lr),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

x_train = x_train.repeat().shuffle(num_train_examples).batch(batch_size)
x_test = x_test.batch(batch_size)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: lr*10**epoch
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=2,
    mode='auto'
)

history = model.fit(
    x_train,
    epochs=epochs,
    steps_per_epoch=num_train_examples//batch_size,
    validation_data=x_test,
    validation_steps=num_test_examples//batch_size,
    callbacks=[early_stop, callbacks]  # early_stop, callbacks, lr_scheduler
)

# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axis([lr, 1e+2, 0, 5])
# plt.show()


def plot_graphs(hist, string, start=0, end=None):
    train_plot = hist.history[string][start:end]
    val_plot = hist.history["val_"+string][start:end]
    epochs_plot = range(len(train_plot))

    plt.plot(epochs_plot, train_plot)
    plt.plot(epochs_plot, val_plot)
    plt.title(string+" and val_"+string)
    plt.figure()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plt.show()

test_loss, test_acc = model.evaluate(x_test, steps=num_test_examples//batch_size)
print(test_acc)
print(test_loss)
