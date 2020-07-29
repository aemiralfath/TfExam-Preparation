import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing import image

(train_examples, val_examples), info = tfds.load(
    'cats_vs_dogs',
    with_info=True,
    as_supervised=True,
    split=['train[:80%]', 'train[80%:]']
)

num_classes = info.features['label'].num_classes
num_training_examples = 0
num_val_examples = 0

for example in train_examples:
    num_training_examples += 1

for example in val_examples:
    num_val_examples += 1

lr = 6e-4
batch_size = 128
epochs = 10
image_size = 150


def format_image(imgori, label):
    imgre = tf.image.resize(imgori, (image_size, image_size))/255.
    return imgre, label


train_batches = train_examples.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = val_examples.map(format_image).batch(batch_size).prefetch(1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=lr),
    loss="binary_crossentropy",
    metrics=['accuracy']
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: lr*(10**(epoch/1))
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=2,
    mode='auto'
)

history = model.fit(
    train_batches,
    steps_per_epoch=num_training_examples//batch_size,
    epochs=epochs,
    validation_data=validation_batches,
    validation_steps=num_val_examples//batch_size,
    callbacks=[early_stop]
)

# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axis([lr, 1e+0, 0, 1])
# plt.show()


def plot_graphs(hist, string, start=0, end=None):
    train_plot = hist.history[string][start:end]
    val_plot = hist.history["val_"+string][start:end]
    epochs_plot = range(len(train_plot))

    plt.plot(epochs_plot, train_plot, label='Training '+string)
    plt.plot(epochs_plot, val_plot, label='Training '+string)
    plt.title(string+" and val_"+string)
    plt.figure()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plt.show()

uploaded = ['dog1.jpg', 'dog2.jpg', 'dog3.jpg']
for fn in uploaded:
    img = image.load_img(os.path.join('dataset/upload/', fn), target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] > 0.5:
        print(fn+" is a dog")
    else:
        print(fn+" is a cat")
