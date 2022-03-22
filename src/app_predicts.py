from keras.models import model_from_json
import numpy as np
import imageio
import cv2

# Carregando o classificador
def load_model():
    a = open('model_cifar-10.json', 'r')
    model_structure = a.read()
    a.close()
    model = model_from_json(model_structure)
    model.load_weights("model_cifar-10.h5")

    return model

def classify(img_path, model):
    img = imageio.imread(img_path)
    predict = model(img)

    return img, predict


model = load_model()

img = imageio.imread('1.png').astype('float32')
img = np.asarray([img])
img /= 255
predict = model.predict(img)
print(predict > 0.5)
