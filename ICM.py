import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import os

# Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data() # 2 tuples which contain dataset of train and test
tf.keras.datasets.cifar10.load_data()

# Get the image classification
classification = ["home", "autombile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck","bed"]

# Convert the labels into a set of 10 numbers to input into the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Normalize the pixels to be values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Create the models architecture
model = Sequential()

# Add the first layer
model.add(Conv2D(32, (5,5), activation="relu", input_shape=(32,32,3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

# Add a pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Add another convolution layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Add a flattening layer
model.add(Flatten())

# Add a layer with 1000 neurons
model.add(Dense(1000, activation="relu"))

# Add a drop out layer
model.add(Dropout(0.5))

# Add a layer with 10 neurons
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
training= int(input("Enter the amount of training you want to do(The more you do the accurate the result): "))
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
hist = model.fit(x_train, y_train_one_hot, batch_size= 256, epochs= training, callbacks=[reduce_lr], validation_split= 0.2)

# Test the model with an example and show it
from PIL import Image
import matplotlib.image as mpimg

name = input("Input the name of the imagine you want to classify with it's included tag(example test.png): ")
print(r"'%s'" % name)
img = mpimg.imread(name)

# Resize the image
from skimage.transform import resize
resized_image = resize(img, (32,32,3))
imgplot = plt.imshow(resized_image)

# Get the models predictions
predictions = model.predict(np.array([resized_image]))
# Show the predictions
predictions

# Sort the predictions from least to greatest
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

# Print the first 5 predictions
for i in range(5):
    print(classification[list_index[i]], ":", round(predictions[0][list_index[i]] * 100, 2),"%")
