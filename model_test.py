import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("./softmax_model")

train_ds = tf.keras.utils.image_dataset_from_directory(
  "./car_color_dataset/train",
  seed = 123,
  image_size = (180, 180),
  batch_size = 32)

test_ds = tf.keras.utils.image_dataset_from_directory(
  "./car_color_dataset/test",
  seed = 123,
  image_size = (180, 180),
  batch_size = 32)

print(model.evaluate(train_ds))



test_image = cv2.imread('./car.jpg')
resize = tf.image.resize(test_image, (180, 180))
prediction = model.predict(np.expand_dims(resize/255, 0))

colors = ["beige", "black", "blue", "brown", "gold", "green", "grey", "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow"]
index_to_colors = dict(zip(range(15), colors))




print(prediction)
print(index_to_colors[np.argmax(prediction, axis=1)[0]])