import os
import zipfile
import tensorflow as tf
# from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = 'dataset/happy-or-sad.zip'
zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall('dataset/happy_sad')
zip_ref.close()

base_dir = '../../dataset/happy_sad'
train_happy_dir = os.path.join(base_dir, 'happy')
train_sad_dir = os.path.join(base_dir, 'sad')

train_happy_names = os.listdir(train_happy_dir)
train_sad_names = os.listdir(train_sad_dir)
print(train_happy_names[:10])
print(train_sad_names[:10])
print(len(train_happy_names))
print(len(train_sad_names))


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > DESIRED_ACCURACY:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


DESIRED_ACCURACY = 0.999
callbacks = MyCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',  # RMSprop(lr=0.001)
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_sad_names+train_happy_names)//10,
    epochs=15,
    verbose=1,
    callbacks=[callbacks]
)
