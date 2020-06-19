import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
import os

def get_label(name):
    if "peace" in name:
        label = 0
    elif "wave" in name:
        label = 1
    elif "fist" in name:
        label = 2
    elif "thumbsup" in name:
        label = 3
    elif "rad" in name:
        label = 4
    else:
        label = 5
    return label

def load_samples(dir):
    cwd = os.getcwd()
    folder = os.path.join(cwd, dir)
    label = get_label(dir)
    samples = [f for f in os.listdir(folder) if not f.startswith('.')]
    x_train = []
    y_train = []
    for i in range(len(samples)):
        img_path = os.path.join(folder, str(samples[i]))
        img = cv.imread(img_path, 0)
        x_train.append(img)
        y_train.append(label)
    return (x_train, y_train)

def load_data(train_or_test):
    # Access datasets directory (training or testing)
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets')
    path = os.path.join(path, train_or_test)
    # datasets contains each class name
    datasets = [f for f in os.listdir(path) if not f.startswith('.')]

    key = []
    x_data = []
    y_train = []
    for i in enumerate(datasets):
        key.append([i[0], i[1]])
        train_data = load_samples(str('datasets/' + train_or_test + '/' + str(i[1])))
        x_data.append(train_data[0])
        if i[0] == 0:
            y_train = train_data[1]
        else:
            y_train = np.hstack((y_train, train_data[1]))

    x_train = np.vstack((x_data))

    return (x_train, y_train), key

def reshape_data(x, y, key):
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    y = np_utils.to_categorical(y, len(key))
    return x, y

def normalize_data(x):
    x = x.astype('float32')
    x /= 255
    return x

if __name__ == "__main__":
    (x_train, y_train), key_train = load_data('training')
    (x_test, y_test), key_test = load_data('testing')

    # x_train, y_train = reshape_data(x_train, y_train, key)
    # x_test, y_test = reshape_data(x_test, y_test, key)

    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(200, 200)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(6)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    probability_model.save('cnn.h5')