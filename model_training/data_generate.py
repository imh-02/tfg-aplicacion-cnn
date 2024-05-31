import numpy as np
from PIL import Image
import os

"""
Variables para especificar el número de imágenes que se van a generar
"""
number_of_images = 450

"""
Ruta donde se va a guardar el dataset generado
"""
datasetPathRoot = "datasetDNI"

"""
Ruta del dataset origen de las imágenes de los dígitos y letras
"""
originDatasetPathRootDigits = "datasetSave12\\digits\\"
originDatasetPathRootLetters = "datasetSave12\\letters\\"


letters_dic = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def obtainLetter(dni):
    numero = int(dni)
    resto = numero % 23
    letras = "TRWAGMYFPDXBNJZSQVHLCKE"
    return letras[resto]

# obtener las rutas en el siguiente nivel respecto a la ruta root
def obteinRoutes(root):
    routes = []
    for i in os.listdir(root):
        routes.append(root + "\\" + i)
    return routes


def generateDataset():

    if not os.path.exists(datasetPathRoot):
        os.makedirs(datasetPathRoot)


    for i in range(number_of_images):

        dniNumbers = np.random.randint(0, 10, 8)

        dniLetterCorresponding = obtainLetter("".join(str(x) for x in dniNumbers))
        dniLetter = letters_dic.index(dniLetterCorresponding)

        resultDNIstring = "".join(str(x) for x in dniNumbers) + dniLetterCorresponding

        generatedNumbers = []

        for j in dniNumbers:

            digitsImagesDirPath = originDatasetPathRootDigits + str(j) + "\\"
            imagesDir = obteinRoutes(digitsImagesDirPath)

            numbersPixels = []

            for k in imagesDir:
                numbersPixels.append(np.array(Image.open(k)))

            randomNumberPosition = np.random.randint(0, len(numbersPixels))
            generatedNumbers.append(numbersPixels[randomNumberPosition])

        lettersPixels = []
        lettersImagesDirPath = originDatasetPathRootLetters + letters_dic[dniLetter].upper() + "\\"
        imagesDir = obteinRoutes(lettersImagesDirPath)
        for k in imagesDir:
            lettersPixels.append(np.array(Image.open(k)))
        
        for j in range(len(generatedNumbers)):
            generatedNumbers[j].shape = (28, 28)
        
        
        randomLetterPosition = np.random.randint(0, len(lettersPixels))
        letter = lettersPixels[randomLetterPosition]
        letter.shape = (28, 28)
        combined_array = np.hstack((generatedNumbers[0], generatedNumbers[1], generatedNumbers[2], generatedNumbers[3], 
                                                  generatedNumbers[4], generatedNumbers[5], 
                                                  generatedNumbers[6], generatedNumbers[7], 
                                                  letter)).astype(np.uint8)
        resultDNIPil = Image.fromarray(combined_array)
        resultDNIPil.save(datasetPathRoot + "\\" + resultDNIstring + ".png")

    return

generateDataset()