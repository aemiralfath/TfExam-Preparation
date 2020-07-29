import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2

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

lr = 1e-3
batch_size = 128
epochs = 3
image_size = 224


def format_image(imgori, label):
    imgre = tf.image.resize(imgori, (image_size, image_size))/255.
    return imgre, label


train_batches = train_examples.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = val_examples.map(format_image).batch(batch_size).prefetch(1)

pre_trained = MobileNetV2(input_shape=(image_size, image_size, 3))
pre_trained.trainable = False
last_output = pre_trained.layers[-2].output
x = layers.Flatten()(last_output)
x = layers.Dense(1, activation='sigmoid')(x)
model = Model(pre_trained.input, x)
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
