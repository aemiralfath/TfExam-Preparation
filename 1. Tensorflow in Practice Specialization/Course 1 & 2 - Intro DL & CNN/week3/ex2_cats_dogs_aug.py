import os
import zipfile
import random
import shutil
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_zip = '../../dataset/kagglecatsanddogs_3367a.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('../../dataset/')
zip_ref.close()


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


try:
    base_dir = '../../dataset/cats-v-dogs'
    make_dir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    make_dir(train_dir)
    make_dir(test_dir)

    cats_train_dir = os.path.join(train_dir, 'cats')
    cats_test_dir = os.path.join(test_dir, 'cats')
    dogs_train_dir = os.path.join(train_dir, 'dogs')
    dogs_test_dir = os.path.join(test_dir, 'dogs')
    make_dir(cats_train_dir)
    make_dir(cats_test_dir)
    make_dir(dogs_train_dir)
    make_dir(dogs_test_dir)
except OSError:
    pass


def split_data(source, training, testing, split):
    data = os.listdir(source)
    random_data = random.sample(data, len(data))
    train_size = len(data)*split

    for i, filename in enumerate(random_data):
        filepath = os.path.join(source, filename)
        if os.path.getsize(filepath) > 0:
            if i < train_size:
                copyfile(filepath, os.path.join(training, filename))
            else:
                copyfile(filepath, os.path.join(testing, filename))
        else:
            print(filename + " is zero length, so ignoring.")


cat_source_dir = '../../dataset/PetImages/Cat/'
dog_source_dir = '../../dataset/PetImages/Dog/'

split_size = .9
split_data(cat_source_dir, cats_train_dir, cats_test_dir, split_size)
split_data(dog_source_dir, dogs_train_dir, dogs_test_dir, split_size)

print(len(os.listdir(cats_train_dir)))
print(len(os.listdir(dogs_train_dir)))
print(len(os.listdir(cats_test_dir)))
print(len(os.listdir(dogs_test_dir)))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

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

train_generator = train_datagen.flow_from_directory(
    '../../dataset/cats_and_dogs_filtered/train',
    batch_size=32,
    class_mode='binary',
    target_size=(150, 150)
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    '../../dataset/cats_and_dogs_filtered/validation',
    batch_size=32,
    class_mode='binary',
    target_size=(150, 150)
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//32,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.n//32
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title("Training and validation loss")
plt.show()

uploaded = ['dog1.jpg', 'dog2.jpg', 'dog3.jpg']
for fn in uploaded:
    img = image.load_img(os.path.join('../../dataset/upload/', fn), target_size=(150, 150))
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
