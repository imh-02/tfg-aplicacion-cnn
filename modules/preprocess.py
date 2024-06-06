from PIL import Image, ImageOps, ImageFilter
import cv2
from torchvision import transforms
import numpy as np
import statistics

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
    heigth = image.size[1]
    width = image.size[0]

    image = ImageOps.pad(image, (width,heigth + 4), color='white')
    image = ImageOps.pad(image, (width + 4,heigth + 4), color='white')
    image = ImageOps.pad(image, (128,128), color='white')
    image = image.point(lambda p: p > 200 and 255)
    image_np = np.array(image)
    image_np = cv2.bitwise_not(image_np)
    image = Image.fromarray(image_np)

    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    image = image.resize((28,28), Image.BICUBIC)
    image = image.convert('L')

    image_preprocessed = image

    transform = transforms.Compose([transforms.ToTensor()])
    image_preprocessed = transform(image)

    return image_preprocessed




  
    