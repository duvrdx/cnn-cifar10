from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization



# Carregando e dividindo dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f'Exemplos de treino: {x_train.shape[0]}')
print(f'Exemplos de teste: {x_test.shape[0]}')

# Convertendo em float32 e normalizando
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train /= 255
x_test /= 255

# Hot encoding
class_train = np_utils.to_categorical(y_train, 10)
class_test  = np_utils.to_categorical(y_test, 10)



#---------------------------------------------------------------#
# Modelando rede neural
model = Sequential()

# Adicionando camada de convolução
model.add(Conv2D(64, (3,3), input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Adicionando camada de convolução
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Camadas ocultas
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))

# Camada de saída
model.add(Dense(units=10, activation='softmax'))

#---------------------------------------------------------------#
# Compilando rede neural
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

# Treinando rede neural
model.fit(x_train, class_train, batch_size=128, epochs=100, validation_data=[x_test, class_test])


#----------------------------------------------------------------#
# Salvando classificador e pesos dos neurônios
model_json = model.to_json()
with open("model_cifar-10.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_cifar-10.h5")
