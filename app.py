import src.app_predicts as cifar10
import matplotlib.pyplot as plt
import imageio

def main():
    # Variaveis
    flag = 'exit'
    kb_input = ''
    img_path = ''
    img_plt = ''
    img = ''

    # Carregando modelo de rede neural
    model = cifar10.load_model()

    # Loop de execução
    while kb_input != flag:
        img_path = str(input('Insira o caminho da imagem que deseja realizar a predição:\n'))
        while img_path == '':
            img_path = str(input('Caminho inválido! Digite novamente:\n'))
        
        # Convertendo imagem para o padrão da rede neural
        img = cifar10.convert_img(img_path)

        # Realizando predição
        predict = cifar10.classify(img,model)

        # Mostrando resultados
        img_plt = imageio.imread(img_path)
        plt.imshow(img_plt)
        plt.title(cifar10.wich_class(predict))
        plt.show()

        # Verificando
        kb_input = str(input('Deseja tentar com outra imagem? Caso não, digite "exit":\n'))

if __name__ == '__main__':
    main()