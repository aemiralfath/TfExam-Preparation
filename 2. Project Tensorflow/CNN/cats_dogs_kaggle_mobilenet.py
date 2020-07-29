import os
import zipfile
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile

import tensorflow as tf
from keras import layers
from keras import Model
from keras.preprocessing import image
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator

local_zip = "dataset/kagglecatsanddogs_3367a.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('dataset/')
zip_ref.close()


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


try:
    base_dir = 'dataset/cats-v-dogs'
    make_dir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    make_dir(train_dir)
    make_dir(val_dir)

    cats_train_dir = os.path.join(train_dir, 'cats')
    cats_val_dir = os.path.join(val_dir, 'cats')
    dogs_train_dir = os.path.join(train_dir, 'dogs')
    dogs_val_dir = os.path.join(val_dir, 'dogs')
    make_dir(cats_train_dir)
    make_dir(cats_val_dir)
    make_dir(dogs_train_dir)
    make_dir(dogs_val_dir)
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


cat_source_dir = 'dataset/PetImages/Cat/'
dog_source_dir = 'dataset/PetImages/Dog/'

split_size = .9
split_data(cat_source_dir, cats_train_dir, cats_val_dir, split_size)
split_data(dog_source_dir, dogs_train_dir, dogs_val_dir, split_size)

print(len(os.listdir(cats_train_dir)))
print(len(os.listdir(dogs_train_dir)))
print(len(os.listdir(cats_val_dir)))
print(len(os.listdir(dogs_val_dir)))

lr = 1e-3
batch_size = 128
epochs = 3
image_size = 224

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
    train_dir,
    batch_size=batch_size,
    class_mode='binary',
    target_size=(image_size, image_size)
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    batch_size=batch_size,
    class_mode='binary',
    target_size=(image_size, image_size)
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
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
