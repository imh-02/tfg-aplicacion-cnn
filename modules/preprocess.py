from PIL import Image, ImageOps
import cv2
from torchvision import transforms
import numpy as np

"""
Función para convertir en imágenes cada una de las páginas del documento PDF con la entrada de datos
"""
def convert_page_to_image(page):

    pix = page.get_pixmap()  # render page to an image
    image_data = pix.samples  # get raw image bytes
    img = Image.frombytes("RGB", [pix.width, pix.height], image_data) ## Convierte la imagen a escala de grises

    return img

"""
Función de preprocesamiento de la imagen, mediante la binarización de la misma, para mejorar la tarea de OCR
"""
def binarize_image(image, threshold):
    # Convierte la imagen en blanco y negro utilizando un umbral fijo
    image_result = image.convert('L').point(lambda p: 255 if p > threshold else 0, '1')
    image_result = image.convert('RGB')
    return image_result

def preprocess_CNN(image):
    image = image.convert('L')
    
    image = ImageOps.pad(image, (7,22), color='white')
    image = ImageOps.pad(image, (28,28), color='white')

    image = image.point(lambda p: p > 190 and 255)

    image = Image.eval(image, lambda p: 255 - p)

    transform = transforms.Compose([transforms.ToTensor()])
    image_preprocessed = transform(image)
    return image_preprocessed




  
    