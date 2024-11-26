# Authors: Andre Costa Barros, Seamus Delaney 
# Description: This script is based on example code provided for training a Convolutional Neural Network (CNN) on the MNIST dataset using TensorFlow and Keras.

# Constants
EPOCHS = 17
SPLIT = 0.2
SHUFFLE = True
BATCH = 32
OPT = 'RMSprop'

import numpy as np
import tensorflow
import matplotlib.pyplot as plt

# Random Number Generator
np.random.seed(1)                    # Initialise system RNG.
tensorflow.random.set_seed(2)        # and the seed of the Tensorflow backend.

VERSION = tensorflow.__version__
print(VERSION)                       # Print TensorFlow version. Should be at least 2.0.
NEW_KERAS = True if int(VERSION.split('.')[1]) >= 12 else False  # Check if using a modern (> 2.12) Keras

# Keras libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense                    # Fully-connected layer
if NEW_KERAS: from tensorflow.keras.layers import InputLayer # InputLayer if using a modern (> 2.12) Keras
from tensorflow.keras.layers import Conv2D                   # 2-d Convolutional layer
from tensorflow.keras.layers import MaxPooling2D             # 2-d Max-pooling layer
from tensorflow.keras.layers import Flatten                  # Converts 2-d layer to 1-d layer
from tensorflow.keras.layers import Activation               # Nonlinearities
from tensorflow.keras.layers import Dropout                  # Dropout layer for regularization

from tensorflow.keras.utils import to_categorical            # Utility for one-hot encoding

from tensorflow.keras.datasets import mnist                  # MNIST dataset

# Load MNIST dataset
(training_inputs, training_labels), (testing_inputs, testing_labels) = mnist.load_data()

print(training_inputs.shape, training_inputs.dtype, testing_inputs.shape, testing_inputs.dtype)

# Preprocess the data
training_images = (training_inputs.astype('float32')/255)[:,:,:,np.newaxis]  # Normalize and reshape training images
categorical_training_outputs = to_categorical(training_labels)               # One-hot encode training labels
testing_images = (testing_inputs.astype('float32')/255)[:,:,:,np.newaxis]    # Normalize and reshape testing images
categorical_testing_outputs = to_categorical(testing_labels)                 # One-hot encode testing labels

print(training_images.shape, training_images.dtype)
print(testing_images.shape, testing_images.dtype)
print(categorical_training_outputs.shape, training_labels.shape)
print(categorical_testing_outputs.shape, testing_labels.shape)

# Plot some training images
plt.figure(figsize=(14,4))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(training_images[i,:,:,0], cmap='gray')  # Display image in grayscale
    plt.title(str(training_labels[i]))                 # Display label as title
    plt.axis('off')                                    # Hide axes

from tensorflow.keras.layers import Dense

in_shape = training_images.shape[1:]   # Returns a 2-tuple (row, cols) which is the shape of the input images.

# Build the model
model = Sequential()

# 3 Convolution layers, 2 Classification layers, Dropout layer after each layer
if NEW_KERAS:
    model.add(InputLayer(shape=in_shape))  # Input layer
    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=in_shape, activation='relu'))  # First Conv layer
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  # First MaxPooling layer
    model.add(Dropout(0.25))  # Dropout layer for regularization

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))  # Second Conv layer
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  # Second MaxPooling layer
    model.add(Dropout(0.25))  # Dropout layer for regularization

    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Third Conv layer
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  # Third MaxPooling layer
    model.add(Dropout(0.4))  # Dropout layer for regularization
else:
    model.add(InputLayer(shape=in_shape))  # Input layer
    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=in_shape, activation='relu'))  # First Conv layer
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  # First MaxPooling layer
    model.add(Dropout(0.25))  # Dropout layer for regularization

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))  # Second Conv layer
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  # Second MaxPooling layer
    model.add(Dropout(0.25))  # Dropout layer for regularization

    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Third Conv layer
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  # Third MaxPooling layer
    model.add(Dropout(0.4))  # Dropout layer for regularization

model.add(Flatten())  # Flatten the 3D output to 1D
model.add(Dense(512, activation='relu'))  # Fully-connected layer with ReLU activation
model.add(Dropout(0.4))  # Dropout layer for regularization
model.add(Dense(10, activation='softmax'))  # Output layer with softmax activation for classification

print("The Keras network model")
model.summary()  # Print model summary

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-3), metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

# Early stopping callback
stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
                     verbose=2, mode='auto',
                     restore_best_weights=True)

# Train the model
history = model.fit(training_images, categorical_training_outputs,
                    epochs=EPOCHS,
                    batch_size=BATCH,
                    shuffle=SHUFFLE,
                    validation_split=SPLIT,
                    verbose=2,
                    callbacks=[stop])

# Plot training and validation losses
plt.figure('Training and Validation Losses per epoch', figsize=(8,8))
plt.plot(history.history['loss'], label='training')  # Training data error per epoch.
plt.plot(history.history['val_loss'], label='validation')  # Validation error per epoch.
plt.grid(True)
plt.legend()
plt.xlabel('Epoch Number')
plt.ylabel('Loss')

# Evaluate the model on the testing set
print("Performance of network on testing set:")
test_loss, test_acc = model.evaluate(testing_images, categorical_testing_outputs)
print("Accuracy on testing data: {:6.2f}%".format(test_acc*100))
print("Test error (loss):        {:8.4f}".format(test_loss))

# Print performance metrics
print("Performance of network:")
print("Accuracy on training data:   {:6.2f}%".format(history.history['accuracy'][-1]*100))
print("Accuracy on validation data: {:6.2f}%".format(history.history['val_accuracy'][-1]*100))
print("Accuracy on testing data:    {:6.2f}%".format(test_acc*100))

plt.show()