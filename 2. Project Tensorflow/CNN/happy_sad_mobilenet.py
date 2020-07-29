import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = "dataset/happy-or-sad.zip"
zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall('dataset/happy_sad')
zip_ref.close()

base_dir = "dataset/happy_sad"
happy_dir = os.path.join(base_dir, 'happy')
sad_dir = os.path.join(base_dir, 'sad')

happy_names = [os.path.join(happy_dir, fn) for fn in os.listdir(happy_dir)]
happy_label = ['0' for fn in os.listdir(happy_dir)]
sad_names = [os.path.join(sad_dir, fn) for fn in os.listdir(sad_dir)]
sad_label = ['1' for fn in os.listdir(happy_dir)]

train_happy_size = int(len(happy_names)*0.75)
train_sad_size = int(len(sad_names)*0.75)

x_train = pd.Series(happy_names[:train_happy_size]+sad_names[:train_sad_size])
y_train = pd.Series(happy_label[:train_happy_size]+sad_label[:train_sad_size])

x_test = pd.Series(happy_names[train_happy_size:]+sad_names[train_sad_size:])
y_test = pd.Series(happy_label[train_happy_size:]+sad_label[train_sad_size:])

print(x_train[:10])
print(y_train[:10])
print(len(x_train))
print(len(y_train))

print(x_test[:10])
print(y_test[:10])
print(len(x_test))
print(len(y_test))

train_df = {'filename': x_train, 'label': y_train}
train_df = pd.DataFrame(train_df)
train_df.head(5)

test_df = {'filename': x_test, 'label': y_test}
test_df = pd.DataFrame(test_df)
test_df.head(5)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


lr = 1e-3
epochs = 10
batch_size = 10
image_size = 224
callbacks = MyCallback()

pre_trained = MobileNetV2(input_shape=(image_size, image_size, 3))
pre_trained.trainable = False
last_output = pre_trained.layers[-2].output
x = layers.Flatten()(last_output)
x = layers.Dense(1, activation='sigmoid')(x)
model = Model(pre_trained.input, x)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=lr),
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
    rescale=1/255.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    shuffle=True,
    seed=42,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

val_datagen = ImageDataGenerator(rescale=1/255.)
val_generator = val_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
    callbacks=[early_stop, callbacks]  # lr_scheduler, early_stop, callbacks
)

# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axis([lr, 1e+1, 0, 5])
# plt.show()


def plot_graphs(hist, string, start=0, end=None):
    train_plot = hist.history[string][start:end]
    val_plot = hist.history["val_"+string][start:end]
    epochs_plot = range(len(train_plot))

    plt.plot(epochs_plot, train_plot)
    plt.plot(epochs_plot, val_plot)
    plt.title(string+" and val_"+string)
    plt.figure()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plt.show()
