import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2

import matplotlib.pylab as plt
import numpy as np
import logging
import time

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    with_info=True,
    as_supervised=True,
    split=['train[:80%]', 'train[80%:]']
)

num_classes = info.features['label'].num_classes
num_training_examples = 0
num_validation_examples = 0

for example in train_examples:
    num_training_examples += 1

for example in validation_examples:
    num_validation_examples += 1

IMAGE_RES = 224
BATCH_SIZE = 32


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.
    return image, label


train_batches = train_examples.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

feature_extractor = MobileNetV2(input_shape=(IMAGE_RES, IMAGE_RES, 3))
feature_extractor.trainable = False  # freeze layer
last_output = feature_extractor.layers[-2].output  # remove last layer
x = layers.Flatten()(last_output)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)
model = Model(feature_extractor.input, x)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 3
history = model.fit(
    train_batches,
    epochs=EPOCHS,
    validation_data=validation_batches
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

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

class_names = np.array(info.features['label'].names)
print(class_names)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

feature_batch = feature_extractor(image_batch)
print(feature_batch.shape)

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = []
for res in predicted_batch:
    predicted_ids.append(1 if res > 0.5 else 0)

predicted_ids = np.array(predicted_ids).reshape(label_batch.shape)
predicted_class_names = class_names[predicted_ids]
print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10, 9))
for n in range(30):
    plt.subplot(6, 5, n+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(), color=color)
    plt.axis('off')
plt.show()

t = time.time()
export_path_keras = "{}.h5".format(int(t))
model.save(export_path_keras)
