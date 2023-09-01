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

#gabor kernels
k1 = np.array([[[[-6.69151833e-01, -3.14653923e-01, -3.14653923e-01,
          -5.37584976e-01,  5.37584976e-01]],

        [[-7.82317751e-01, -6.33730689e-01, -5.14434187e-01,
          -7.17917408e-01,  1.31089228e-01]],

        [[-5.31019723e-01, -8.08255440e-01, -6.87005324e-01,
          -8.07730225e-01, -3.55373356e-01]],

        [[ 5.70748167e-17, -7.15350798e-01, -7.94203518e-01,
          -7.94203518e-01, -7.15350798e-01]],

        [[ 5.31019723e-01, -3.55373356e-01, -8.07730225e-01,
          -6.87005324e-01, -8.08255440e-01]],

        [[ 7.82317751e-01,  1.31089228e-01, -7.17917408e-01,
          -5.14434187e-01, -6.33730689e-01]],

        [[ 6.69151833e-01,  5.37584976e-01, -5.37584976e-01,
          -3.14653923e-01, -3.14653923e-01]]],


       [[[-6.95807811e-01, -5.14434187e-01, -6.33730689e-01,
          -1.31089228e-01,  7.17917408e-01]],

        [[-8.13481747e-01, -7.80276573e-01, -7.80276573e-01,
          -4.11812858e-01,  4.11812858e-01]],

        [[-5.52173144e-01, -8.16930689e-01, -8.43606698e-01,
          -6.48807646e-01, -8.08243032e-02]],

        [[ 5.93484189e-17, -5.56503892e-01, -8.00280670e-01,
          -8.00280670e-01, -5.56503892e-01]],

        [[ 5.52173144e-01, -8.08243032e-02, -6.48807646e-01,
          -8.43606698e-01, -8.16930689e-01]],

        [[ 8.13481747e-01,  4.11812858e-01, -4.11812858e-01,
          -7.80276573e-01, -7.80276573e-01]],

        [[ 6.95807811e-01,  7.17917408e-01, -1.31089228e-01,
          -6.33730689e-01, -5.14434187e-01]]],


       [[[-7.12308418e-01, -6.87005324e-01, -8.08255440e-01,
           3.55373356e-01,  8.07730225e-01]],

        [[-8.32772911e-01, -8.43606698e-01, -8.16930689e-01,
           8.08243032e-02,  6.48807646e-01]],

        [[-5.65267553e-01, -7.13119367e-01, -7.13119367e-01,
          -2.23720815e-01,  2.23720815e-01]],

        [[ 6.07558261e-17, -3.04820902e-01, -5.04703516e-01,
          -5.04703516e-01, -3.04820902e-01]],

        [[ 5.65267553e-01,  2.23720815e-01, -2.23720815e-01,
          -7.13119367e-01, -7.13119367e-01]],

        [[ 8.32772911e-01,  6.48807646e-01,  8.08243032e-02,
          -8.16930689e-01, -8.43606698e-01]],

        [[ 7.12308418e-01,  8.07730225e-01,  3.55373356e-01,
          -8.08255440e-01, -6.87005324e-01]]],


       [[[-7.17895122e-01, -7.94203518e-01, -7.15350798e-01,
           7.15350798e-01,  7.94203518e-01]],

        [[-8.39304430e-01, -8.00280670e-01, -5.56503892e-01,
           5.56503892e-01,  8.00280670e-01]],

        [[-5.69701001e-01, -5.04703516e-01, -3.04820902e-01,
           3.04820902e-01,  5.04703516e-01]],

        [[ 6.12323400e-17,  6.12323400e-17,  6.12323400e-17,
           6.12323400e-17,  6.12323400e-17]],

        [[ 5.69701001e-01,  5.04703516e-01,  3.04820902e-01,
          -3.04820902e-01, -5.04703516e-01]],

        [[ 8.39304430e-01,  8.00280670e-01,  5.56503892e-01,
          -5.56503892e-01, -8.00280670e-01]],

        [[ 7.17895122e-01,  7.94203518e-01,  7.15350798e-01,
          -7.15350798e-01, -7.94203518e-01]]],


       [[[-7.12308418e-01, -8.07730225e-01, -3.55373356e-01,
           8.08255440e-01,  6.87005324e-01]],

        [[-8.32772911e-01, -6.48807646e-01, -8.08243032e-02,
           8.16930689e-01,  8.43606698e-01]],

        [[-5.65267553e-01, -2.23720815e-01,  2.23720815e-01,
           7.13119367e-01,  7.13119367e-01]],

        [[ 6.07558261e-17,  3.04820902e-01,  5.04703516e-01,
           5.04703516e-01,  3.04820902e-01]],

        [[ 5.65267553e-01,  7.13119367e-01,  7.13119367e-01,
           2.23720815e-01, -2.23720815e-01]],

        [[ 8.32772911e-01,  8.43606698e-01,  8.16930689e-01,
          -8.08243032e-02, -6.48807646e-01]],

        [[ 7.12308418e-01,  6.87005324e-01,  8.08255440e-01,
          -3.55373356e-01, -8.07730225e-01]]],


       [[[-6.95807811e-01, -7.17917408e-01,  1.31089228e-01,
           6.33730689e-01,  5.14434187e-01]],

        [[-8.13481747e-01, -4.11812858e-01,  4.11812858e-01,
           7.80276573e-01,  7.80276573e-01]],

        [[-5.52173144e-01,  8.08243032e-02,  6.48807646e-01,
           8.43606698e-01,  8.16930689e-01]],

        [[ 5.93484189e-17,  5.56503892e-01,  8.00280670e-01,
           8.00280670e-01,  5.56503892e-01]],

        [[ 5.52173144e-01,  8.16930689e-01,  8.43606698e-01,
           6.48807646e-01,  8.08243032e-02]],

        [[ 8.13481747e-01,  7.80276573e-01,  7.80276573e-01,
           4.11812858e-01, -4.11812858e-01]],

        [[ 6.95807811e-01,  5.14434187e-01,  6.33730689e-01,
           1.31089228e-01, -7.17917408e-01]]],


       [[[-6.69151833e-01, -5.37584976e-01,  5.37584976e-01,
           3.14653923e-01,  3.14653923e-01]],

        [[-7.82317751e-01, -1.31089228e-01,  7.17917408e-01,
           5.14434187e-01,  6.33730689e-01]],

        [[-5.31019723e-01,  3.55373356e-01,  8.07730225e-01,
           6.87005324e-01,  8.08255440e-01]],

        [[ 5.70748167e-17,  7.15350798e-01,  7.94203518e-01,
           7.94203518e-01,  7.15350798e-01]],

        [[ 5.31019723e-01,  8.08255440e-01,  6.87005324e-01,
           8.07730225e-01,  3.55373356e-01]],

        [[ 7.82317751e-01,  6.33730689e-01,  5.14434187e-01,
           7.17917408e-01, -1.31089228e-01]],

        [[ 6.69151833e-01,  3.14653923e-01,  3.14653923e-01,
           5.37584976e-01, -5.37584976e-01]]]])


#Define custom kernels
def custom_layer(shape, dtype=None):
  print(shape)
  kernel_in = k1
  kernel = tf.constant(kernel_in, dtype=tf.float32)
  return kernel

model = models.Sequential()

model.add(layers.Conv2D(5, (7, 7), activation='relu', kernel_initializer=custom_layer, trainable=False, padding='same', l2(1e-4), input_shape=(200, 200, 1))) #
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
model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.4))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='sgd',
    metrics=['accuracy', tf.keras.metrics.AUC()]
    )

# fit model
checkpoint = keras.callbacks.ModelCheckpoint('saved_mdoel{epoch:04d}.h5', save_freq='epoch', period=500)
h = model.fit(train_generator, epochs = 2000, validation_data=validation_generator, callbacks=[checkpoint])
model.save_weights('gab-a.h5')
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

ax = df[['accuracy', 'val_accuracy']].plot(figsize=(8, 5))
ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Accuracies', fontsize=16)
ax.set_title('Model Accuracies', fontsize=17)
ax.legend(['train_acc', 'val_acc'], fontsize=12)
ax.figure.savefig('train_.pdf', dpi=150)
ax.figure.savefig('train_.png', dpi=150)

ax = df[['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot(figsize=(8, 5))
ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Losses ~ Accuracies', fontsize=16)
ax.set_title('Model Performance', fontsize=17)
ax.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], fontsize=12)
ax.figure.savefig('comb_.pdf', dpi=150)
ax.figure.savefig('comb_.png', dpi=150)

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

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

disp.plot(cmap=plt.cm.Blues)
fig1 = plt.gcf()
fig1.delaxes(fig1.axes[1])
plt.show()