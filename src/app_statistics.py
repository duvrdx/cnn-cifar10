from os import stat
import statistics as stats
import app_predicts as cifar
import numpy as np

import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, classification_report

# Carregando data e model
x_train, x_test, class_train, class_test = stats.load_cifar()
model = cifar.load_model('data/model_cifar-10.json','data/model_cifar-10.h5')

# Carregando histórico da rede neural
with open('statistics/history', 'rb') as file:
    history = pickle.load(file)

# Imprimindo e salvando histórico
stats.plotmodelhistory(history)
print(f'Accuracy: {history.history["val_accuracy"][-1]}')
print(f'Loss: {history.history["val_loss"][-1]}')

# Imprimindo heatmap
pred = model.predict(x_test)
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

y_pred_classes = np.argmax(pred, axis = 1)
y_true = np.argmax(class_test, axis = 1)
errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = pred[errors]
y_true_errors = y_true[errors]
x_test_errors = x_test[errors]

cm = confusion_matrix(y_true, y_pred_classes)
thresh = cm.max() / 2,

fig, ax = plt.subplots(figsize = (12, 12))
im, cbar = stats.heatmap(cm, labels, labels, ax = ax,
cmap = plt.cm.Blues, cbarlabel = 'count of predictions')

texts = stats.annotate_heatmap(im, data = cm, threshold = thresh)

fig.tight_layout()
plt.show()
fig.savefig('statistics/heatmap.png', dpi=72)
