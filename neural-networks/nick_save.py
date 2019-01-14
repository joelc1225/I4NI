#you might get an error here if you don't have the necessary libraries installed in your environment.  See prerequisites comment above.
import tensorflow as tf
import numpy as np
import random
import scipy.ndimage
import matplotlib.pyplot as plt

#print tensorflow version just to verify that we have successfully installed tensorflow and it is running
print(tf.__version__);

#import mnist data.  it is part of the tensorflow package so if you have sucessfully installed tensorflow, you should have access to it
#dimensions: 28 pixels x 28 pixels
#contents: hand-written digits between 0-9
mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

#unpack the data and put in variables for train data and test data
(cuda_education_train, y_train), (cuda_education_test, y_test) = mnist.load_data()

#print out multidimensional array. this is the raw data that the computer sees.
#the numbers represent pixel values.
#notice that you can see an outline of a digit in the raw data representation
print("print out raw data\n\n")
print(cuda_education_train[7])


# #show the human-friendly version of the image, WITH COLOR
# plt.imshow(cuda_education_train[7])
# plt.title("show the human-friendly version of the image, WITH COLOR")
# plt.show()


# #show the human-friendly version of the image, WITHOUT COLOR
# plt.imshow(cuda_education_train[7], cmap = plt.cm.binary)
# plt.title("show the human-friendly version of the image, WITHOUT COLOR")
# plt.show()

#normalize data
cuda_education_train = tf.keras.utils.normalize(cuda_education_train, axis=1)
cuda_education_test = tf.keras.utils.normalize(cuda_education_test, axis=1)


#print out multidimensional array
#this time the data should be normalized
print("print out normalized version of data. notice the pixel values are much smaller\n\n")
print(cuda_education_train[7])

# #show the human-friendly version of the image WITH COLOR
# #this is the normalized version
# plt.title("show the NORMALIZED version of the image WITH COLOR")
# plt.imshow(cuda_education_train[7])
# plt.show()

# #show the human-friendly version of the image WITHOUT COLOR
# #this is the normalized version
# plt.title("show the NORMALIZED version of the image WITHOUT COLOR")
# plt.imshow(cuda_education_train[7], cmap = plt.cm.binary)
# plt.show()

#initiate a model of type sequential
model = tf.keras.models.Sequential()

#flatten the tensor.  for more information, visit https://youtu.be/mFAIBMbACMA
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#hidden layer with activation function of relu. for more information on activation functions, visit https://youtu.be/-7scQpJT7uo
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#hidden layer with activation function of relu
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#output layer. we use softmax as the output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#compile model using specific parameters for optimization, loss and the metric we want to track
model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(cuda_education_train,y_train,epochs=2)

val_loss, val_acc = model.evaluate(cuda_education_test, y_test)
#print out loss and accuracy values
print(val_loss, val_acc)


model.save('cuda_education_handwritten_digit_reader.model')
new_model = tf.keras.models.load_model('cuda_education_handwritten_digit_reader.model')


predictions = new_model.predict([cuda_education_test])
#print(predictions)

# #print out the prediction for the thirty sixth digit in the test data.
# print("\n\n what does the neural network think the 36th handwritten digit in the TEST DATA is")
# print(np.argmax(predictions[35]))
# print("\n\n")

# #lets see what the thirty sixth digit in the test data looks like
# plt.title("the first digit in the test data")
# plt.imshow(cuda_education_test[35])
# plt.show()


#read the image from the file we specified
# y = scipy.ndimage.imread("im_09.jpg", flatten=True)
#print out the image in RGB array form
# print("show submitted image as array of pixels")
# print(y)

# #show the image we submitted
# plt.imshow(y)
# plt.title("show submitted image")
# plt.show()
# print("\n\n")

# #show the image when we invert the pixels
# print("show the image when we invert the pixels")
y = np.vectorize(lambda x: 255 - x)(scipy.ndimage.imread("im_03.jpg", flatten=True))
# plt.imshow(y)
# print(y)
# plt.title("show inverted pixel image")
# plt.show()
# print("\n\n")




# #show a flattened version of the image in pixel array form
# print("show a flattened version of the image in pixel array form")
# z= np.ndarray.flatten(scipy.ndimage.imread("im_09.jpg", flatten=True))
# print(z)
# print("\n\n")


# #let us combine all the operations and put everything into variable x
# #takes all the numbers in the array, does 255 - each number and gives a result.  it is inverting the pixels.
# x = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("im_09.jpg", flatten=True)))
# print(x)
# #plt.imshow(x)
# #plt.show()



#x = tf.keras.utils.normalize(x)
#print(x)
#plt.imshow(x, cmap = plt.cm.binary)
#plt.show()

#x = tf.keras.utils.normalize(x)
#print(x.shape)
#print(x)

#make sure to reshape the image so that the predict function will not throw an error
y = np.array(y).reshape(1,28,28)
#print(x.shape)


predictions = new_model.predict([y])
#what does the neural network think the handwritten digit is
print("\n\nwhat does the neural network think the handwritten digit is")
print(np.argmax(predictions[0]))
