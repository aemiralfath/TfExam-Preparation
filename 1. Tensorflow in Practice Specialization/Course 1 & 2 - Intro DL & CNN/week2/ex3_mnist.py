import tensorflow as tf
import tensorflow_datasets as tfds

import logging
import numpy as np
import matplotlib.pyplot as plt

tfds.disable_progress_bar()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print('Number of training examples: {}'.format(num_train_examples))
print('Number of test examples:     {}'.format(num_test_examples))


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# train_dataset = train_dataset.cache()
# test_dataset = test_dataset.cache()

plt.figure()
for image, label in test_dataset.take(1):
    image = image.numpy().reshape((28, 28))
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    break

plt.show()

plt.figure(figsize=(10, 10))
index = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, index+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    index += 1

plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=num_train_examples//BATCH_SIZE)

test_loss, test_accuracy = model.evaluate(test_dataset, steps=num_test_examples//32)
print('Accuracy on test dataset:', test_accuracy)
print('Accuracy on test loss:', test_loss)

for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    print('shape:', predictions.shape)
    print('result:', predictions[0])
    print('predict:', np.argmax(predictions[0]))
    print('real:', test_labels[0])


def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_labels, img_plot = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img_plot[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_labels:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[int(predicted_label)],
                                         100*np.max(predictions_array),
                                         class_names[true_labels]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def custom_plot(i, predictions_array, true_labels, images):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions_array, true_labels, images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions_array, true_labels)
    plt.show()


custom_plot(0, predictions, test_labels, test_images)
custom_plot(12, predictions, test_labels, test_images)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for index in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*index+1)
    plot_image(index, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*index+2)
    plot_value_array(index, predictions, test_labels)
plt.show()

img = test_images[0]
print(img.shape)

img = np.array([img])
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)
print(np.argmax(predictions_single[0]))

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
