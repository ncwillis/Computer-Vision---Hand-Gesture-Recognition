import cv2 as cv
import os

def initialize_dataset():
    """
    :return: Index number of image sample, path of image sample, and directory folder
    """
    # Create dataset directory to store files
    print("Enter Dataset Name and press Enter: ")
    dir = input()

    cwd = os.getcwd()
    path = os.path.join(cwd, "datasets")
    if "test" in dir:
        path = os.path.join(path, 'testing')
    else:
        path = os.path.join(path, 'training')
    path = os.path.join(path, dir)
    if not os.path.exists(path):
        os.mkdir(path)
    i = len(os.listdir(path)) + 1
    return i, path, dir

def save_sample(dir, i, path, mask):
    """
    :param dir: Directory name (string)
    :param i: Sample index number (integer)
    :param path: File path of sample
    :param mask: Sample image
    :return: File name (string)
    """
    file_name = str(dir + '_' + str(i) + '.jpg')
    cv.imwrite(os.path.join(path, file_name), mask)
    return file_name