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
    image = cv2.resize(input_image, (95,20))

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para obtener una imagen binaria
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # binary = cv2.bitwise_not(binary)

    # kernel = np.ones((1,1), np.uint8)
    # image_dilation = cv2.dilate(binary, kernel=kernel, iterations=1)

    # not_image_dilation = cv2.bitwise_not(image_dilation)

    image = Image.fromarray(binary)

    # image.show()
    # input("press")
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir una dimensión para el batch
    return image

  
    