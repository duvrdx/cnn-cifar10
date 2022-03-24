from keras.models import model_from_json
import numpy as np
import imageio
import cv2

# Converte a imagem para um formato em que o classificador entenda
def convert_img(img):
    img = imageio.imread('1.png').astype('float32')
    img = np.asarray([img])
    img /= 255

    return img

# Carrega o modelo de rede neural
def load_model(path_model, path_weights):
    a = open(path_model, 'r')
    model_structure = a.read()
    a.close()
    model = model_from_json(model_structure)
    model.load_weights(path_weights)

    return model

# Detecta qual é a classe da predição pra uma forma categorica
def wich_class(array):
    pattern = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for pos, item in enumerate(array>0.5):
        if item:
            return pattern[pos]
            
# Classifica uma imagem utilizando a rede neural
def classify(img, model):
    predict = model.predict(img)

    return img, predict