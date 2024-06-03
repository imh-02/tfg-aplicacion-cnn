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

"""
Transformación utilizada durante el entrenamiento de la red CNN
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Función para procesar la imagen
def preprocess_imageCNN(image_pil):

    # img = np.array(image_pil)
    # # convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # threshold
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # # invert
    # thresh = 255 - thresh

    # # apply horizontal morphology close
    # kernel = np.ones((5 ,191), np.uint8)
    # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # # get external contours
    # contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]

    # # draw contours
    # result = img.copy()
    # for cntr in contours:
    #     # get bounding boxes
    #     pad = 10
    #     x,y,w,h = cv2.boundingRect(cntr)
    #     cv2.rectangle(result, (x-pad, y-pad), (x+w+pad, y+h+pad), (0, 0, 255), 4)
    
    # # recorta la imagen a partir de los contornos
    # x,y,w,h = cv2.boundingRect(contours[0])
    # image = Image.fromarray(img[y:y+h, x:x+w])

    # image.show()
    # input("press")

    # image_np = np.array(image)
    # # input_image = np.array(image_pil)
    # image = cv2.resize(image_np, (95,20))
    # image = Image.fromarray(image)
    # height = image_pil.size[1]
    # image = ImageOps.pad(image_pil, (95, height), color='white')
    # image = ImageOps.pad(image, (95, 20), color='white')


    input_image = np.array(image_pil)

    input_image = cv2.resize(input_image, (95, 20))
    # # # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # # Aplicar umbral para obtener una imagen binaria
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # binary = cv2.bitwise_not(binary)

    # kernel = np.ones((1,1), np.uint8)
    # image_dilation = cv2.dilate(binary, kernel=kernel, iterations=1)

    # not_image_dilation = cv2.bitwise_not(image_dilation)

    image = Image.fromarray(binary)

    image = image.convert('L')

    # image.show()
    # input("press")
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir una dimensión para el batch
    return image

  
    