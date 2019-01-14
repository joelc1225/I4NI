import tensorflow as tf
import numpy as np
import random
import scipy.ndimage
import matplotlib.pyplot as plt


new_model = tf.keras.models.load_model('cuda_education_handwritten_digit_reader.model')




#import mnist data.  it is part of the tensorflow package so if you have sucessfully installed tensorflow, you should have access to it
#dimensions: 28 pixels x 28 pixels
#contents: hand-written digits between 0-9
mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

#unpack the data and put in variables for train data and test data
(cuda_education_train, y_train), (cuda_education_test, y_test) = mnist.load_data()


predictions = new_model.predict([cuda_education_test])
#print(predictions)

#print out the prediction for the thirty sixth digit in the test data.
print("\n\n what does the neural network think the 36th handwritten digit in the TEST DATA is")
print(np.argmax(predictions[35]))
print("\n\n")

#lets see what the thirty sixth digit in the test data looks like
plt.title("the 36th digit in the test data")
plt.imshow(cuda_education_test[35])
plt.show()

"""
#read the image from the file we specified
y = scipy.ndimage.imread("im_09.jpg", flatten=True)
#print out the image in RGB array form
print("show submitted image as array of pixels")
print(y)

#show the image we submitted
plt.imshow(y)
plt.title("show submitted image")
plt.show()
print("\n\n")

#show the image when we invert the pixels
print("show the image when we invert the pixels")
y = np.vectorize(lambda x: 255 - x)(scipy.ndimage.imread("im_09.jpg", flatten=True))
plt.imshow(y)
print(y)
plt.title("show inverted pixel image")
plt.show()
print("\n\n")




#show a flattened version of the image in pixel array form
print("show a flattened version of the image in pixel array form")
z= np.ndarray.flatten(scipy.ndimage.imread("im_09.jpg", flatten=True))
print(z)
print("\n\n")


#let us combine all the operations and put everything into variable x
#takes all the numbers in the array, does 255 - each number and gives a result.  it is inverting the pixels.
x = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("im_04.jpg", flatten=True)))
print(x)
#plt.imshow(x)
#plt.show()



#x = tf.keras.utils.normalize(x)
#print(x)
#plt.imshow(x, cmap = plt.cm.binary)
#plt.show()

#x = tf.keras.utils.normalize(x)
#print(x.shape)
#print(x)

#make sure to reshape the image so that the predict function will not throw an error
x = np.array(x).reshape(1,28,28)
#print(x.shape)


predictions = new_model.predict([x])
#what does the neural network think the handwritten digit is
print("\n\nwhat does the neural network think the handwritten digit is")
print(np.argmax(predictions[0]))
"""
