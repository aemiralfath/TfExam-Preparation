import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

IMAGE_RES = 224
model = tf.keras.Sequential([
    MobileNetV2(input_shape=(IMAGE_RES, IMAGE_RES, 3))
])

grace_hopper = tf.keras.utils.get_file(
    'image.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
)
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))
grace_hopper = np.array(grace_hopper)/255.
print(grace_hopper.shape)

result = model.predict(grace_hopper[np.newaxis, ...])
print(result.shape)

predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = decode_predictions(result, top=1)[0][0][1]
_ = plt.title(f"Prediction: {predicted_class_name}")
plt.show()
