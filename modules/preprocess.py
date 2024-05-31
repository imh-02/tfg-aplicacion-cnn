from PIL import Image
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

"""
Transformación utilizada durante el entrenamiento de la red CNN
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Función para procesar la imagen
def preprocess_imageCNN(image_pil):
    input_image = np.array(image_pil)
    image = cv2.resize(input_image, (252, 28))
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir una dimensión para el batch
    return image

  
    