import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt


(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()

model = Sequential([
    Conv2D(filters=6, kernel_size=3, strides=1, activation='tanh', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(filters=16, kernel_size=3, strides=1, activation='tanh'),
    MaxPooling2D(pool_size=2, strides=2),
    Flatten(),
    Dense(units=120, activation='sigmoid'),
    Dense(units=84, activation='sigmoid'),
    Dense(units=10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(training_images/255.0, to_categorical(training_labels), epochs=10, batch_size=32, validation_split = 0.2)

score = model.evaluate(test_images/255.0, to_categorical(test_labels), batch_size=32)

model.save('lenetcoef.h5')

print("Testset Loss: %f" % score[0])
print("Testset Accuracy: %f" % score[1])

plt.figure()

plt.subplot(211)
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.grid(True)

plt.subplot(212)
plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.show()