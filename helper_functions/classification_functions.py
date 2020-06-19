import numpy as np

def get_class(label):
    """
    :param label: Integer value corresponding to class
    :return: Class name for given label value
    """
    if label == 0:
        hand = "peace"
    elif label == 1:
        hand = "wave"
    elif label == 2:
        hand = "fist"
    elif label == 3:
        hand = "thumbsup"
    elif label == 4:
        hand = "rad"
    else:
        hand = "ok"
    return hand

def process_sample(img):
    """
    :param img: Sample image
    :return: Processed sample image with expanded dimensions
    """
    img = img.astype('float32')
    # expand dimensions
    img = (np.expand_dims(img, 0))
    return img

def get_prediction(sample, model):
    """
    :param sample: Sample image
    :param model: Machine learning model object
    :return: Predicted class name
    """
    prediction = model.predict(sample)
    prediction = np.argmax(prediction[0])
    class_name = get_class(prediction)
    return class_name