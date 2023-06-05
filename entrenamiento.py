# Importamos las librerías necesarias para el análisis de imágenes
import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import red as network
import os

# Se definen funciones auxiliares
# Función para detectar si una imagen está corrupta y no puede ser analizada
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

# Función para normalizar la forma de la matriz
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data
# Función para transformar cadenas en listas
def split(l, n):
    for i in range(0,len(l), n):
        yield l[i:i+n]
# Función propia del lenguaje que permite mostrar las imágenes en
# pantalla, comparando la imagen de entrenamiento, entrada y salida
def plot(data, test, predicted):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    if not os.path.exists('resultados'):
        os.mkdir('resultados')
    count=0
    file_count=0
    for d in split(data, 4):
        if not len(d)is 1: 
            fig, axarr = plt.subplots(len(d), 3)
            for i in range(len(d)):
                if i==0:
                    axarr[i, 0].set_title('Imagen de Entrenamiento')
                    axarr[i, 1].set_title("Imagen de Entrada")
                    axarr[i, 2].set_title('Imagen de Salida')

                axarr[i, 0].imshow(data[count])
                axarr[i, 0].axis('off')
                axarr[i, 1].imshow(test[count])
                axarr[i, 1].axis('off')
                axarr[i, 2].imshow(predicted[count])
                axarr[i, 2].axis('off')
                count = count+1

            plt.tight_layout()
            plt.savefig("resultados/resultado_"+str(file_count)+".png")
            file_count=file_count+1
            plt.show()
        else:
            fig, axarr = plt.subplots(1, 3)
            axarr[0].set_title('Imagen de Entrenamiento')
            axarr[1].set_title("Imagen de Entrada")
            axarr[2].set_title('Imagen de Salida')

            axarr[0].imshow(data[count])
            axarr[0].axis('off')
            axarr[1].imshow(test[count])
            axarr[1].axis('off')
            axarr[2].imshow(predicted[count])
            axarr[2].axis('off')
            count = count+1
            plt.tight_layout()
            plt.savefig("resultados/resultado_"+str(file_count)+".png")
            file_count=file_count+1
            plt.show()

# Función de preprocesamiento
def preprocessing(img, w=128, h=128):
    # Se redimensiona la imagen de entrada a 128x128 px
    img = resize(img, (w,h), mode='reflect')

    # Se setea el umbral en unidades binarias (1 a -1)
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1

    # Se guarda la imagen procesada
    flatten = np.reshape(shift, (w*h))
    return flatten

# Función principal
def main():
    # Se importan las imágenes
    import cv2
    import glob
    img_dir = "entrenamiento/" # Se lee el directorio que contiene las imágenes
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    data = []
    # Se transforma a B/N y se ajusta el brillo
    for f1 in files:
        img = rgb2gray(cv2.imread(f1))
        data.append(img)
        
    # Preprocesamiento de las imágenes
    print("Se inicia el preprocesado de las imágenes...")
    data = [preprocessing(d) for d in data]

    # Se genera la red neuronal de Hopfield
    model = network.HopfieldNetwork()
    model.train_weights(data)

    # Se leen las imágenes contra las que se probará el modelo
    img_dir = "prueba/"
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    test = []

    for f1 in files:
        img = rgb2gray(cv2.imread(f1))
        test.append(img)
    test = [preprocessing(d) for d in test]
    
    # Se muestran los resultados en pantalla
    predicted = model.predict(test, threshold=0, asyn=False)
    print("Mostrando resultados del análisis...")
    plot(data, test, predicted)

if __name__ == '__main__':
    main()