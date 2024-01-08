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
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization

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
        # shuffle=True
	)

validation_generator = validation_datagen.flow_from_directory(
        './validation',
        target_size = (200, 200),
        color_mode = 'grayscale',
        batch_size = batch_size,
        class_mode='binary',
        # shuffle=True
	)

test_generator = test_datagen.flow_from_directory(
        './test',
        target_size = (200, 200),
        color_mode = 'grayscale',
        batch_size = batch_size,
        class_mode='binary',
        # shuffle=True
	)

k1 = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1]
    ])

kx = k1.reshape(7,7,1,1)

def custom_layer(shape, dtype=None):
  print(shape)
  kernel_in = kx
  kernel = tf.constant(kernel_in, dtype=tf.float32)
  return kernel

model = models.Sequential()

model.add(layers.Conv2D(1, (7, 7), activation='relu', kernel_initializer=custom_layer, trainable=False, padding='same', input_shape=(200, 200, 1))) 
model.add(layers.MaxPool2D(pool_size=(2,2), padding='same', strides=(2,2)))

model.add(layers.Conv2D(512, (7, 7), activation='relu', padding='same', kernel_regularizer = regularizers.l2(1e-4)))
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
model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.4))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='sgd',
    metrics=['accuracy', tf.keras.metrics.AUC()]
    )

# fit model
checkpoint = keras.callbacks.ModelCheckpoint('saved_model{epoch:04d}.h5', period=500)
h = model.fit(train_generator, epochs = 2000, validation_data=validation_generator, callbacks=[checkpoint])
model.save_weights('l1-04_apr-2.h5')
_, acc, auc = model.evaluate(validation_generator, steps=len(validation_generator), verbose=1)
print('ACC:', acc * 100.0)
print('AUC:', auc * 100.0)


ax = df[['loss','val_loss']].plot(figsize=(8,5), xlabel='Epochs', ylabel='Accuracies', title='Model Losses', legend=(['train_loss', 'val_loss']))
# ax = df.plot(figsize=(8,5), xlabel='Epochs', ylabel='Accuracies', title='Model Accuracies', legend=(['Train', 'Validation']))
ax.figure.savefig('loss_.pdf', dpi=150)
ax.figure.savefig('loss_.png', dpi=150)
ax.figure.savefig('losses', dpi=150)

ax = df[['accuracy','val_accuracy']].plot(figsize=(8,5), xlabel='Epochs', ylabel='Accuracies', title='Model Accuracies', legend=(['train_acc', 'val_acc']))
# ax = df.plot(figsize=(8,5), xlabel='Epochs', ylabel='Accuracies', title='Model Accuracies', legend=(['Train', 'Validation']))
ax.figure.savefig('train_.pdf', dpi=150)
ax.figure.savefig('train_.png', dpi=150)
ax.figure.savefig('train', dpi=150)

ax = df[['loss','val_loss','accuracy','val_accuracy']].plot(figsize=(8,5), xlabel='Epochs', ylabel='Losses ~ Accuracies', title='Model performance', legend=(['train_loss', 'val_loss', 'train_acc', 'val_acc']))
# ax = df.plot(figsize=(8,5), xlabel='Epochs', ylabel='Accuracies', title='Model Accuracies', legend=(['Train', 'Validation']))
ax.figure.savefig('comb_.pdf', dpi=150)
ax.figure.savefig('comb_.png', dpi=150)
ax.figure.savefig('comb', dpi=150)

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


tr_results = model.evaluate(train_generator)
print('eval_train: ',tr_results)

va_results = model.evaluate(validation_generator)
print('eval_validation: ', va_results)

te_results = model.evaluate(test_generator)
print('eval_test: ', te_results)

## CM libraries

from sklearn.metrics import confusion_matrix
from scipy.special import expit

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

"""## CM Train"""

true_classes = train_generator.classes
class_labels = list(train_generator.class_indices.keys())

y_true = train_generator.classes
y_pred = (model.predict(train_generator).ravel()>0.5)

TP = tf.math.count_nonzero(y_pred * y_true)
TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
FP = tf.math.count_nonzero(y_pred * (y_true - 1))
FN = tf.math.count_nonzero((y_pred - 1) * y_true)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

tf.print('tr-pre', precision)
tf.print('tr-rec', recall)
tf.print('tr-f1', f1)

cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

disp.plot(cmap=plt.cm.Blues)
fig1 = plt.gcf()
fig1.delaxes(fig1.axes[1])
plt.show()
plt.draw()
# fig1.savefig('cm_train.pdf', dpi=100)
# fig1.savefig('cm_train', dpi=150)

"""## CM Validation"""

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

y_true = validation_generator.classes
y_pred = (model.predict(validation_generator).ravel()>0.5)

TP = tf.math.count_nonzero(y_pred * y_true)
TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
FP = tf.math.count_nonzero(y_pred * (y_true - 1))
FN = tf.math.count_nonzero((y_pred - 1) * y_true)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

tf.print('v-pre', precision)
tf.print('v-rec', recall)
tf.print('v-f1', f1)

cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

disp.plot(cmap=plt.cm.Blues)
fig1 = plt.gcf()
fig1.delaxes(fig1.axes[1])
plt.show()
plt.draw()
# fig1.savefig('cm_valid.pdf', dpi=100)
# fig1.savefig('cm_valid', dpi=150)

"""## CM TEST"""

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

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

disp.plot(cmap=plt.cm.Blues)
fig1 = plt.gcf()
fig1.delaxes(fig1.axes[1])
plt.show()
plt.draw()
# fig1.savefig('cm_test.pdf', dpi=100)
# fig1.savefig('cm_test', dpi=150)
