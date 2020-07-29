import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

pic_index = 0
fig = plt.gcf()
fig.set_size_inches(16, 16)

pic_index += 8
next_horse_pic = [os.path.join(train_horse_dir, fname)
                  for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pic = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pic+next_human_pic):
    sp = plt.subplot(4, 4, i+1)
    sp.axis('off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

lr = 1e-3
train_batch = 128
val_batch = 32
epochs = 15
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
    patience=3,
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
    target_size=(300, 300),
    batch_size=train_batch,
    class_mode='binary'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(300, 300),
    batch_size=val_batch,
    class_mode='binary'
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//train_batch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n//val_batch,
    callbacks=[early_stop]
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
    img = image.load_img(path, target_size=(300, 300))
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
