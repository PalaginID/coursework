import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils.np_utils import to_categorical
from keras import models
from keras.layers import Bidirectional, Dense, Flatten, Dropout, LSTM
import matplotlib.pyplot as plt
from keras.datasets import cifar10


(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
training_images = training_images.reshape(50000, 1024, 3)
test_images = test_images.reshape(10000, 1024, 3)
training_images = training_images/255.0
test_images = test_images/255.0

model = models.Sequential([
        Bidirectional(LSTM(32, input_shape=(1024, 3), return_sequences=True)),
        Dropout(0.25),
        Flatten(),
        Dense(10, activation='softmax')
        ])
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=["accuracy"])

history = model.fit(training_images, to_categorical(training_labels), batch_size = 32, epochs=10, validation_split = 0.2)

score = model.evaluate(test_images, to_categorical(test_labels), batch_size=32)

model.save('lstmcoef.h5')

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

