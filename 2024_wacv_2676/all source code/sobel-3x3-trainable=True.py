import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os

import keras
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
# from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

def getDevices():
    from tensorflow import config
    devices = [device for device in config.list_physical_devices() if "GPU" == device.device_type]
    devices = ["/gpu:{}".format(i) for i, device in enumerate(devices)]
    return devices

devices = getDevices()
print(devices)

batch_size = 128

from keras.preprocessing.image import ImageDataGenerator

## this is the augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest') 

validation_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest')

## Generators
train_generator = train_datagen.flow_from_directory(
        './train',
        target_size=(200, 200),
        color_mode = 'grayscale',
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
        './validation',
        target_size = (200, 200),
        color_mode = 'grayscale',
        batch_size = batch_size,
        class_mode='binary',
        shuffle=True)

test_generator = test_datagen.flow_from_directory(
        './test',
        target_size = (200, 200),
        color_mode = 'grayscale',
        batch_size = batch_size,
        class_mode='binary',
        shuffle=True)

## Defining the SOBEL Kernels
s_hori = np.array([
        [ -1,  0,  1],
        [ -2,  0,  2],
        [ -1,  0,  1]
    ])

s_vert = np.array([
        [ -1, -2, -1],
        [  0,  0,  0],
        [  1,  2,  1]
    ])

sh = s_hori.reshape(3,3,1,1)
sv = s_vert.reshape(3,3,1,1)

ks = np.concatenate((sh, sv), axis=3)

#Define custom kernels
def custom_layer(shape, dtype=None):
  print(shape)
  kernel_in = ks
  kernel = tf.constant(kernel_in, dtype=tf.float32)
  return kernel

model = models.Sequential()

model.add(layers.Conv2D(2, (3, 3), activation='relu', kernel_initializer=custom_layer, trainable=True, padding='same', input_shape=(200, 200, 1)))
model.add(layers.MaxPool2D(pool_size=(2,2), padding='same', strides=(2,2)))

model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(1e-4)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dropout(0.4))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='sgd',
    metrics=['accuracy', tf.keras.metrics.AUC()]
    )

# fit model
checkpoint = keras.callbacks.ModelCheckpoint('saved_mdoel{epoch:04d}.h5', save_freq='epoch', period=500)
h = model.fit(train_generator, epochs = 2000, validation_data=validation_generator, callbacks=[checkpoint])
model.save_weights('l2-may6.h5')
_, acc, auc = model.evaluate(validation_generator, steps=len(validation_generator), verbose=1)
print('ACC:', acc * 100.0)
print('AUC:', auc * 100.0)


ax = df[['loss', 'val_loss']].plot(figsize=(8, 5))
ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Losses', fontsize=16)
ax.set_title('Model Losses', fontsize=17)
ax.legend(['train_loss', 'val_loss'], fontsize=12)
ax.figure.savefig('loss.pdf', dpi=150)
ax.figure.savefig('loss.png', dpi=150)
# ax.figure.savefig('losses', dpi=150)

ax = df[['accuracy', 'val_accuracy']].plot(figsize=(8, 5))
ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Accuracies', fontsize=16)
ax.set_title('Model Accuracies', fontsize=17)
ax.legend(['train_acc', 'val_acc'], fontsize=12)
ax.figure.savefig('train_.pdf', dpi=150)
ax.figure.savefig('train_.png', dpi=150)
# ax.figure.savefig('train', dpi=150)

ax = df[['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot(figsize=(8, 5))
ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Losses ~ Accuracies', fontsize=16)
ax.set_title('Model Performance', fontsize=17)
ax.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], fontsize=12)
ax.figure.savefig('comb_.pdf', dpi=150)
ax.figure.savefig('comb_.png', dpi=150)
# ax.figure.savefig('comb', dpi=150)

scores = model.evaluate(train_generator, verbose=0)
results = model.evaluate(validation_generator, verbose = 0)
test_results = model.evaluate(test_generator, verbose = 0)
best_acc = max(h.history['accuracy']); best_loss = min(h.history['loss']); best_val_acc = max(h.history['val_accuracy']); best_val_loss = min(h.history['val_loss'])

print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')
print(f'Training Score : {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%;  \n' )
print(f'Validation Score: {model.metrics_names[0]} of {results[0]}; validation {model.metrics_names[1]} of {results[1]*100}%;  \n' )
print(f'Test Score: test {model.metrics_names[0]} of {test_results[0]}; test {model.metrics_names[1]} of {test_results[1]*100}%;  \n' )
print(f' Best Accuracy: {best_acc*100}%;  Best Loss: {best_loss}; Best Val_Accuracy: {best_val_acc*100}%; Best Val_Loss: {best_val_loss} \n' )
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')


"""### example for precision recall f1
https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
"""

tr_results = model.evaluate(train_generator)
print('eval_train: ',tr_results)

va_results = model.evaluate(validation_generator)
print('eval_validation: ', va_results)

te_results = model.evaluate(test_generator)
print('eval_test: ', te_results)

"""## CM TEST"""
from scipy.special import expit
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

y_true = test_generator.classes
y_pred = (model.predict(test_generator).ravel()>0.5)

import tensorflow as tf
TP = tf.math.count_nonzero(y_pred * y_true)
TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
FP = tf.math.count_nonzero(y_pred * (y_true - 1))
FN = tf.math.count_nonzero((y_pred - 1) * y_true)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

tf.print('test-pre', precision)
tf.print('test-rec', recall)
tf.print('test-f1', f1)

cm = confusion_matrix(y_true, y_pred)
print(cm)