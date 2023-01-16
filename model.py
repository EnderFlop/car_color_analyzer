import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#following https://www.tensorflow.org/tutorials/load_data/images#load_data_using_a_keras_utility

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  "./car_color_dataset/train",
  seed = 123,
  image_size = (img_height, img_width),
  batch_size = batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  "./car_color_dataset/test",
  seed = 123,
  image_size = (img_height, img_width),
  batch_size = batch_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 15

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(180,180,3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, (3,3), 1, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.sparse_categorical_crossentropy,
  metrics=['accuracy'])

print(model.summary())
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

trained_model = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=3,
  callbacks=[tensorboard_callback]
)

print(model.evaluate())

model.save('./softmax_model')