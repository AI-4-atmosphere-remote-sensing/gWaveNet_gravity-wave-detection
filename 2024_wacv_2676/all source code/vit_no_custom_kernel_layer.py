import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os

from tensorflow.keras.layers import Layer, Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Resizing, RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

def getDevices():
    from tensorflow import config
    devices = [device for device in config.list_physical_devices() if "GPU" == device.device_type]
    devices = ["/gpu:{}".format(i) for i, device in enumerate(devices)]
    return devices

devices = getDevices()
print(devices)


# embedding.py
class Patches(Layer):
    def __init__(self, patch_size):
        """ Patches
            Parameters
            ----------
            patch_size: int
                size of a patch (P)
        """
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        """ Pass images to get patches
            Parameters
            ----------
            images: tensor,
                images from dataset
                shape: (..., W, H, C). Example: (64, 32, 32, 3)
            Returns
            -------
            patches: tensor,
                patches extracted from images
                shape: (..., S, P^2 x C) with S = (HW)/(P^2) Example: (64, 64, 48)
        """
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )

        dim = patches.shape[-1]

        patches = tf.reshape(patches, (batch_size, -1, dim))
        return patches


class PatchEmbedding(Layer):
    def __init__(self, patch_size, image_size, projection_dim):
        """ PatchEmbedding
            Parameters
            ----------
            patch_size: int
                size of a patch (P)
            image_size: int
                size of a image (H or W)
            projection_dim: D
                size of project dimension before passing patches through transformer
        """
        super(PatchEmbedding, self).__init__()

        # S = self.num_patches: Number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # cls token used for last mlp network
        self.cls_token = self.add_weight(
            "cls_token",
            shape=[1, 1, projection_dim],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )
        self.patches = Patches(patch_size)

        self.projection = Dense(units=projection_dim)

        # self.position_embedding shape: (..., S + 1, D)
        self.position_embedding = self.add_weight(
            "position_embeddings",
            shape=[self.num_patches + 1, projection_dim],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )

    def call(self, images):
        """ Pass images to embed position information
            Parameters
            ----------
                        images: tensor,
                images from dataset
                shape: (..., W, H, C). Example: (64, 32, 32, 3)
            Returns
            -------
            encoded_patches: tensor,
                embed patches with position information and concat with cls token
                shape: (..., S + 1, D) with S = (HW)/(P^2) Example: (64, 65, 768)
        """

        # Get patches from images
        # patch shape: (..., S, NEW_C)
        patch = self.patches(images)

        # encoded_patches shape: (..., S, D)
        encoded_patches = self.projection(patch)

        batch_size = tf.shape(images)[0]

        hidden_size = tf.shape(encoded_patches)[-1]

        # cls_broadcasted shape: (..., 1, D)
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls_token, [batch_size, 1, hidden_size]),
            dtype=images.dtype,
        )

        # encoded_patches shape: (..., S + 1, D)
        encoded_patches = tf.concat([cls_broadcasted, encoded_patches], axis=1)

        # encoded_patches shape: (..., S + 1, D)
        encoded_patches = encoded_patches + self.position_embedding

        return encoded_patches

# encoder.py

class MLPBlock(Layer):
    def __init__(self, hidden_layers, dropout=0.1, activation='gelu'):
        """ MLP Block in Transformer Encoder

            Parameters
            ----------
            hidden_layers: Python array
                list of layers for mlp block
            dropout: float,
                dropout rate of mlp block
            activation: string
                activation of mlp layer
        """
        super(MLPBlock, self).__init__()

        layers = []
        for num_units in hidden_layers:
            layers.extend([
                Dense(num_units, activation=activation),
                Dropout(dropout)
            ])

        self.mlp = Sequential(layers)

    def call(self, inputs, *args, **kwargs):
        """ Pass output of multi-head attention to mlp block
            Parameters
            ----------
            inputs: tensor,
                multi-head attention outputs
                shape: (..., S, D). Example: (64, 100, 768)
            Returns
            -------
            outputs: tensor,
                attention + mlp outputs
                shape: (..., S, D). Example: (64, 100, 768)
        """

        outputs = self.mlp(inputs, *args, **kwargs)
        return outputs


class TransformerBlock(Layer):
    def __init__(self, num_heads, D, hidden_layers, dropout=0.1, norm_eps=1e-12):
        """ Transformer blocks which includes multi-head attention layer and mlp block

            Parameters
            ----------
            num_heads: int,
                number of heads of multi-head attention layer
            D: int,
                size of each attention head for value
                        hidden_layers: Python array
                list of layers for mlp block
            dropout: float,
                dropout rate of mlp block
            norm_eps: float,
                eps of layer norm
        """
        super(TransformerBlock, self).__init__()
        # Attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=D, dropout=dropout
        )
        self.norm_attention = LayerNormalization(epsilon=norm_eps)

        # MLP
        self.mlp = MLPBlock(hidden_layers, dropout)
        self.norm_mlp = LayerNormalization(epsilon=norm_eps)

    def call(self, inputs):
        """
            Pass Embedded Patches through the layers
            Parameters
            ----------
            inputs: tensor,
                Embedded Patches
                shape: (..., S, D). Example: (64, 100, 768)
            Returns
            -------
            outputs: tensor,
                attention + mlp outputs
                shape: (..., S, D). Example: (64, 100, 768)
        """
        # Feed attention
        norm_attention = self.norm_attention(inputs)

        attention = self.attention(query=norm_attention, value=norm_attention)

        # Skip Connection
        attention += inputs

        # Feed MLP
        outputs = self.mlp(self.norm_mlp(attention))

        # Skip Connection
        outputs += attention

        return outputs


class TransformerEncoder(Layer):
    def __init__(self, num_layers, num_heads, D, mlp_dim, dropout=0.1, norm_eps=1e-12):
        """
            Transformer Encoder which comprises several transformer layers
            Parameters
            ----------
            num_layers: int,
                number of transformer layers
                Example: 12
            num_heads: int,
                number of heads of multi-head attention layer
            D: int
                size of each attention head for value
            mlp_dim:
                mlp size or dimension of hidden layer of mlp block
            dropout: float,
                dropout rate of mlp block
            norm_eps: float,
                eps of layer norm
        """
        super(TransformerEncoder, self).__init__()

        # Create num_layers of TransformerBlock
        self.encoder = Sequential(
            [
                TransformerBlock(num_heads=num_heads,
                                 D=D,
                                 hidden_layers=[mlp_dim, D],
                                 dropout=dropout,
                                 norm_eps=norm_eps)
                for _ in range(num_layers)
            ]
        )

    def call(self, inputs, *args, **kwargs):
        """
            Pass Embedded Patches through the layers
            Parameters
            ----------
            inputs: tensor,
                Embedded Patches
                shape: (..., S, D). Example: (64, 100, 768)
            Returns
            -------
            outputs: tensor,
                attention + mlp outputs
                shape: (..., S, D). Example: (64, 100, 768)
        """
        outputs = self.encoder(inputs, *args, **kwargs)
        return outputs

# vit model.py


class ViT(Model):
    def __init__(self, num_layers=12, num_heads=12, D=768, mlp_dim=3072, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        """
            VIT Model
            Parameters
            ----------
            num_layers: int,
                number of transformer layers
                Example: 12
            num_heads: int,
                number of heads of multi-head attention layer
            D: int
                size of each attention head for value
            mlp_dim:
                mlp size or dimension of hidden layer of mlp block
            num_classes:
                number of classes
            patch_size: int
                size of a patch (P)
            image_size: int
                size of a image (H or W)
            dropout: float,
                dropout rate of mlp block
            norm_eps: float,
                eps of layer norm
        """
        super(ViT, self).__init__()
        # Data augmentation
        self.data_augmentation = Sequential([
            Rescaling(scale=1./255),
            Resizing(image_size, image_size),
            RandomFlip("horizontal"),
            RandomRotation(factor=0.02),
            RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ])

        # Patch embedding
        self.embedding = PatchEmbedding(patch_size, image_size, D)

        # Encoder with transformer
        self.encoder = TransformerEncoder(
            num_heads=num_heads,
            num_layers=num_layers,
            D=D,
            mlp_dim=mlp_dim,
            dropout=dropout,
            norm_eps=norm_eps,
        )

        # MLP head
        self.mlp_head = Sequential([
            LayerNormalization(epsilon=norm_eps),
            Dense(mlp_dim),
            Dropout(dropout),
            Dense(1, activation='sigmoid'), #Dense(num_classes, activation='softmax'),
        ])

        self.last_layer_norm = LayerNormalization(epsilon=norm_eps)

    def call(self, inputs):
        # Create augmented data
        # augmented shape: (..., image_size, image_size, c)
        augmented = self.data_augmentation(inputs)

        # Create position embedding + CLS Token
        # embedded shape: (..., S + 1, D)
        embedded = self.embedding(augmented)

        # Encode patchs with transformer
        # embedded shape: (..., S + 1, D)
        encoded = self.encoder(embedded)

        # Embedded CLS
        # embedded_cls shape: (..., D)
        embedded_cls = encoded[:, 0]

        # Last layer norm
        y = self.last_layer_norm(embedded_cls)

        # Feed MLP head
        # output shape: (..., num_classes)

        output = self.mlp_head(y)

        return output


class ViTBase(ViT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=12,
                         num_heads=12,
                         D=768,
                         mlp_dim=3072,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)


class ViTLarge(ViT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=24,
                         num_heads=16,
                         D=1024,
                         mlp_dim=4096,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)


class ViTHuge(ViT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=32,
                         num_heads=16,
                         D=1280,
                         mlp_dim=5120,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)

model_type = 'custom'  # valid option: custom, base, large, huge
num_classes = 2
patch_size = 40
num_heads=6 # Number of attention heads
att_size=32 # Size of each attention head for value
num_layer=6 #Number of attention layer
mlp_size=1024 # Size of hidden layer in MLP block
lr=0.001 #'Learning rate'
weight_decay=1e-4 #'Weight decay'
batch_size=128 #Batch size
epochs=2000
image_size=200
image_channels=3

valid_folder='./validation'
train_folder='./train'
test_folder= './test'
model_folder='./saved_model'
train_ds = image_dataset_from_directory(
        train_folder,
        seed=123,
        image_size=(image_size, image_size),
        shuffle=True,
        batch_size=batch_size,
	#class_mode='binary',
    	)
val_ds = image_dataset_from_directory(
        valid_folder,
        seed=123,
        image_size=(image_size, image_size),
        shuffle=True,
        batch_size=batch_size,
	#class_mode='binary',
    	)
test_ds = image_dataset_from_directory(
        test_folder,
        seed=123,
        image_size=(image_size, image_size),
        shuffle=True,
        batch_size=batch_size,
	#class_mode='binary',
    	)

if model_type == 'base':
    model = ViTBase()
elif model_type == 'large':
    model = ViTLarge()
elif model_type == 'huge':
    model = ViTHuge()
else:
    model = ViT(
        num_classes=num_classes,
        patch_size=patch_size,
        image_size=image_size,
        num_heads=num_heads,
        D=att_size,
        mlp_dim=mlp_size,
        num_layers=num_layer
    )
model.build(input_shape=(None, image_size,
                         image_size, image_channels))
#optimizer = Adam(learning_rate=lr, weight_decay=weight_decay)
#optimizer = SGD(learning_rate=lr, weight_decay=weight_decay, momentum=0.9)
optimizer = 'sgd'
# loss = SparseCategoricalCrossentropy()
loss='binary_crossentropy',

model.compile(optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.AUC()])

labels = np.array([x[1].numpy() for x in list(train_ds)])
labels

v_labels = np.array([x[1].numpy() for x in list(val_ds)])
v_labels

t_labels = np.array([x[1].numpy() for x in list(test_ds)])
t_labels

h = model.fit(train_ds,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=val_ds,)
model.save(model_folder)

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

scores = model.evaluate(train_ds, verbose=0)
results = model.evaluate(val_ds, verbose = 0)
test_results = model.evaluate(test_ds, verbose = 0)
best_acc = max(h.history['accuracy']); best_loss = min(h.history['loss']); best_val_acc = max(h.history['val_accuracy']); best_val_loss = min(h.history['val_loss'])

print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')
print(f'Training Score : {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%;  \n' )
print(f'Validation Score: {model.metrics_names[0]} of {results[0]}; validation {model.metrics_names[1]} of {results[1]*100}%;  \n' )
print(f'Test Score: test {model.metrics_names[0]} of {test_results[0]}; test {model.metrics_names[1]} of {test_results[1]*100}%;  \n' )
print(f' Best Accuracy: {best_acc*100}%;  Best Loss: {best_loss}; Best Val_Accuracy: {best_val_acc*100}%; Best Val_Loss: {best_val_loss} \n' )
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')

tr_results = model.evaluate(train_ds)
print('eval_train: ',tr_results)

va_results = model.evaluate(val_ds)
print('eval_validation: ', va_results)

te_results = model.evaluate(test_ds)
print('eval_test: ', te_results)

"""## CM TEST"""
from scipy.special import expit
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

true_classes = test_ds.classes
class_labels = list(test_ds.class_indices.keys())

y_true = test_ds.classes
y_pred = (model.predict(test_ds).ravel()>0.5)

import tensorflow as tf
TP = tf.math.count_nonzero(y_pred * y_true)
TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
FP = tf.math.count_nonzero(y_pred * (y_true - 1))
FN = tf.math.count_nonzero((y_pred - 1) * y_true)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

tf.print(f'test-pre {precision:.2f}')
tf.print(f'test-rec {recall:.2f}')
tf.print(f'test-f1 {f1:.2f}')


cm = confusion_matrix(y_true, y_pred)
print(cm)

h.history['accuracy']

h.history['val_accuracy']

predictions = model.predict(test_ds)
print('Result: {}'.format(np.argmax(predictions), axis=1))

predictions

prediction_label = np.where(predictions>=0.5, 0, 1)

print(prediction_label)

test_labels = np.array([x[1].numpy() for x in list(test_ds)])
test_labels = np.reshape(test_labels, (-1,1))
print(test_labels)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(prediction_lable, test_labels)
acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
print('Overall accuracy: {} %'.format(acc*100))
