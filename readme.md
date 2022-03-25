# Projeto de Rede Neural Convolucional - CIFAR-10🧠
Um projeto para praticar os conceitos de deeplearning com redes neurais convolucionais, esse projeto consiste em desenvolver um modelo de rede neural, para classificar imagens em 10 classes diferentes.

Para isso, foi utilizado um *dataset* bem conhecido, que é o [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html).
## Tecnologias utilizadas 💻
- Python 3.7
- Tensorflow 2.0
- Keras
- Numpy
- Pandas
- imageio
- matplotlib

## Navegação entre pastas🖱️
- [data](data)
    > Contém o modelo de rede neural criado pelo arquivo [cnn.py](src/cnn.py)
- [src](src)
    > Contém os arquivos onde é feita a modelagem da rede neural, e outros módulos com funções para o arquivo principal
- [samples](samples)
    > Contém algumas imagens para teste do classificador
- [statistics](statistics)
    > Contém os gráficos gerados pelo script [app_statistics.py](src/app_statistics.py)
- [app.py](app.py)
    > Arquivo principal que roda um app, onde é feita a predição

## Resultados
- Acurácia do Modelo: ***77,88%***
- Valor de Perda: ***0.663***
- [Gráficos](statistics)
### Problemas
- Valor de perda alta
    > Um grande valor de perda, indica que nossa rede neural possui *Overfiting*, que é quando a nossa rede se 'acostuma' com os dados utilizados no treinamento. Uma possível solução possa ser aumentar o número de camadas ocultas, e aumentar o número de épocas. *(Os testes foram realizados com duas camadas ocultas, e 50 épocas, mais informações em [model_cifar10_summary](data/model_cifar10_summary.txt)).*

- Erro na predição de gatos e cachorros
    > Observando a [matriz de confusão](statistics/heatmap.png), conseguimos perceber que há um grande volume de erro entre a predição de gatos e cachorros. A possível solução seria a mesma que a para abaixar o valor de perda. Outra forma seria aumentando o número de dados utilizando o *ImageGenerator* da própria biblioteca.
