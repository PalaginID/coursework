import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import  load_model
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from keras.utils.np_utils import to_categorical

(x_train, y_train), (test_images, test_labels) = cifar10.load_data()

test_labels_1 = keras.utils.to_categorical(test_labels, 10)

test_images = test_images /255.0

print("LeNet-5:")
model_lenet = load_model('lenetcoef.h5')
model_lenet.evaluate(test_images, test_labels_1)
#print(model_lenet.summary())

test_images = test_images.reshape(10000, 1024, 3)

print("Lstm:")
model_lstm = load_model('lstmcoef.h5')
model_lstm.evaluate(test_images, to_categorical(test_labels))
#print(model_lstm.summary())