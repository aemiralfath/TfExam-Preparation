import csv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data(filename):
    with open(filename) as training_file:
        file = csv.reader(training_file)
        images = []
        labels = []

        i = 0
        for row in file:
            if i != 0:
                labels.append(row[0])
                images.append(np.array_split(row[1:785], 28))
            i += 1

        images = np.array(images).astype('float')
        labels = np.array(labels).astype('float')

    return images, labels


train_path = 'dataset/signlanguage_data/sign_mnist_train.csv'
validation_path = 'dataset/signlanguage_data/sign_mnist_test.csv'

train_images, train_labels = get_data(train_path)
validation_images, validation_labels = get_data(validation_path)

print(train_images.shape)
print(train_labels.shape)
print(validation_images.shape)
print(validation_labels.shape)

train_images = np.expand_dims(train_images, axis=3)
validation_images = np.expand_dims(validation_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(
    train_images,
    train_labels,
    batch_size=32
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow(
    validation_images,
    validation_labels,
    batch_size=32
)

print(train_images.shape)
print(validation_images.shape)
print(train_labels)
print(validation_labels)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_images)//32,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=len(validation_images)//32
)

model.evaluate(validation_images, validation_labels)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
