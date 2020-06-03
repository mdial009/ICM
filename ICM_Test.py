# ICM = Image Classification Machine
# Description: This program classifies images

# Import the libraries
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

'''
# Look at the data types of the variables
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# Get the shape of the arrays
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", x_test.shape)
'''
# take a look at the first image as an array
#index = 0
#x_train[index]

# Show the image as a picture
#img = plt.imshow(x_train[index])

# Get the image label
#print("The image label is:", y_train[index])

# Get the image classification
'''
# creating an empty list 
classification = [] 

# number of elemetns as input 
n = int(input("Enter number of elements : ")) 

# iterating till the range 
for i in range(n, classification): 
	ele = input()

	classification.append(ele) # adding the element 
	
print(classification) 
''' 
classification = ["home", "autombile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Print the image class
#print("The image class is:", classification[y_train[index][0]])

# Convert the labels into a set of 10 numbers to input into the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print the new labels
# print(y_train_one_hot)

# Print the new label of the image/picture above
#print("The one hot label is:", y_train_one_hot[index])

# Normalize the pixels to be values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

#x_train[index]

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
'''
# Evaluate the model using the test data set
model.evaluate(x_test, y_test_one_hot)[1]

# Visualize the models accuracy
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper left")
plt.show()

# Visualize the models loss
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper right")
plt.show()
'''

# Test the model with an example and show it
from PIL import Image
import matplotlib.image as mpimg

'''  
im = Image.open("960x0.jpg")  
#new_image = ("home2-300x300.png")
im.imread("960x0.jpg")
plt.imshow(im)
plt.show()

LOGO_FILENAME = ('home.png')
path = os.getcwd()
dir_list = os.listdir(path)
print(dir_list)
for filename in os.listdir(path):
    if (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPEG') or filename == LOGO_FILENAME):
        im = Image.open(filename)
plt.imshow(im)
#plt.show()
'''
name = input("Input the name of the imagine you want to classify with it's included tag(example test.png): ")
print(r"'%s'" % name)
img = mpimg.imread(name)
#print(img)
#imgplot = plt.imshow(img)
#plt.show()

# Resize the image
from skimage.transform import resize
resized_image = resize(img, (32,32,3))
imgplot = plt.imshow(resized_image)
#plt.show()

# Get the models predictions
predictions = model.predict(np.array([resized_image]))
# Show the predictions
#predictions

# Sort the predictions from least to greatest
list_index = [0,1,2,3,4,5,6,7,8,9,10]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

# Show the sorted labels in order
#print(list_index)

# Print the first 5 predictions
for i in range(10):
    print(classification[list_index[i]], ":", round(predictions[0][list_index[i]] * 100, 2),"%")