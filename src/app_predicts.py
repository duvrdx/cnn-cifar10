from keras.models import model_from_json
import numpy as np
import imageio
import cv2

# Converte a imagem para um formato em que o classificador entenda
def convert_img(img_path):
    img = imageio.imread(img_path).astype('float32')
    img = np.asarray([img])
    img /= 255

    return img

# Carrega o modelo de rede neural
def load_model():
    a = open('data/model_cifar-10.json', 'r')
    model_structure = a.read()
    a.close()
    model = model_from_json(model_structure)
    model.load_weights("data/model_cifar-10.h5")

    return model

# Detecta qual é a classe da predição pra uma forma categorica
def wich_class(array):
    pattern = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    return pattern[array.argmax()]
            
# Classifica uma imagem utilizando a rede neural
def classify(img, model):
    predict = model.predict(img)

    return predict