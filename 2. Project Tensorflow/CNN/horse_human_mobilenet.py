import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

local_zip = "dataset/horse-or-human.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('dataset/horse-or-human')

local_zip = "dataset/validation-horse-or-human.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('dataset/validation-horse-or-human')
zip_ref.close()

train_dir = "dataset/horse-or-human"
train_horse_dir = os.path.join(train_dir, 'horses')
train_human_dir = os.path.join(train_dir, 'humans')

val_dir = "dataset/validation-horse-or-human"
val_horse_dir = os.path.join(val_dir, 'horses')
val_human_dir = os.path.join(val_dir, 'humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
validation_horse_names = os.listdir(val_horse_dir)
validation_human_names = os.listdir(val_human_dir)

print('total training horse images:', len(train_horse_names))
print('total training human images:', len(train_human_names))
print('total validation horse images:', len(os.listdir(val_horse_dir)))
print('total validation human images:', len(os.listdir(val_human_dir)))


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True


lr = 1e-1
epochs = 3
image_size = 224
train_batch = 128
val_batch = 32
callbacks = MyCallback()

pre_trained_model = MobileNetV2(input_shape=(image_size, image_size, 3))
pre_trained_model.trainable = False
last_output = pre_trained_model.layers[-2].output
x = layers.Flatten()(last_output)
x = layers.Dense(1, activation='sigmoid')(x)
model = Model(pre_trained_model.input, x)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=lr),  # lr=lr
    loss='binary_crossentropy',
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
    fill_mode='reflect'
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batch,
    class_mode='binary'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=val_batch,
    class_mode='binary'
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//train_batch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n//val_batch,
    callbacks=[early_stop, callbacks]  # early_stop, callbacks, lr_scheduler
)

# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axis([lr, 1e+1, 0, 1])
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

file_test = ['human1.jpg', 'human2.jpg', 'horse1.jpg', 'horse2.jpg']
for fn in file_test:
    path = 'dataset/upload/'+fn
    img = image.load_img(path, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn+" is a human")
    else:
        print(fn+" is a horse")
