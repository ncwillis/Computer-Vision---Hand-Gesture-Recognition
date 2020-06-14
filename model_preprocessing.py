import cv2 as cv
import numpy as np
import keras
import os


def load_samples(dir, label):
    cwd = os.getcwd()
    folder = os.path.join(cwd, dir)
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
        train_data = load_samples(str('datasets/training/'+ str(i[1])), i[0])
        x_data.append(train_data[0])
        if i[0] == 0:
            y_train = train_data[1]
        else:
            y_train = np.hstack((y_train, train_data[1]))

    x_train = np.vstack((x_data))

    return (x_train, y_train), key

if __name__ == "__main__":
    (x_train, y_train), key = load_data('training')
    print(x_train.shape)
    print(y_train.shape)