import torch
from transformers import  AutoModel, AutoProcessor
from routes import signature_model_path, cnn_letters_model_path, cnn_numbers_model_path, device
from modules.preprocess import preprocess_CNN
import cv2
import numpy as np

if device == True:
    device = 'cuda'
else:
    device = 'cpu'

"""
Carga de redes CNN para reconocimiento de caracteres individuales - 2 modelos separados
"""
cnn_letter = torch.load(cnn_letters_model_path).to(device)
cnn_number = torch.load(cnn_numbers_model_path).to(device)

"""
Carga del modelo preentrenado para la clasificación de imágenes de firma, desde el almacenamiento local.
"""
signature_proccessor = AutoProcessor.from_pretrained(signature_model_path)
model_signature = AutoModel.from_pretrained(signature_model_path).to(device)

"""
Diccionario de etiquetas
"""
letters_dict = {
        1: 'A', 
        2: 'B', 
        3: 'C', 
        4: 'D', 
        5: 'E', 
        6: 'F', 
        7: 'G', 
        8: 'H', 
        9: 'I', 
        10: 'J', 
        11: 'K', 
        12: 'L', 
        13: 'M', 
        14: 'N', 
        15: 'O', 
        16: 'P', 
        17: 'Q', 
        18: 'R', 
        19: 'S',
        20: 'T',
        21: 'U',
        22: 'V',
        23: 'W',
        24: 'X',
        25: 'Y',
        26: 'Z'
        }


"""
Función encargada de obtener las coordenadas de las filas de la tabla dado un resultado de detección de celdas, y devolverlas ordenadas.
"""
def getOrderedRow(results):
    # Get the ordered row
    ordered_row = []
    for i in range(len(results['labels'])):
        if results['labels'][i] == 2:
            ordered_row.append(results['boxes'][i].to('cpu'))

    ordered_row.sort(key=lambda x: x[1], reverse=False)

    return ordered_row

"""
Función encargada de obtener las coordenadas de las columnas de la tabla dado un resultado de detección de celdas, y devolverlas ordenadas.
"""
def getOrderedCol(results):

    ordered_col = []
    for i in range(len(results['labels'])):
        if results['labels'][i] == 1:
            ordered_col.append(results['boxes'][i].to('cpu'))
    
    ## Orden por la coordenada x mínima
    ordered_col.sort(key=lambda x: x[0])

    return ordered_col

"""
Función encargada de procesar una fila de la tabla, recortando la imagen del DNI y de la firma, y devolviéndolas.
"""
def processRow(imagen, coordenatesDNI, coordenatesSignature):
    dni_cropped = imagen.crop((coordenatesDNI[0],coordenatesDNI[1], coordenatesDNI[2], coordenatesDNI[3]))
    signature_cropped = imagen.crop((coordenatesSignature[0],coordenatesSignature[1], coordenatesSignature[2], coordenatesSignature[3]))
    return [dni_cropped, signature_cropped]

"""
Función encargada de detectar en una imagen si se encuentra una firma, devolviendo True si la firma es detectada y False en caso contrario.
"""
def detectSignature(image):
    texts = ["signature", "empty"]
    inputs = signature_proccessor(text=texts, images=image, padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model_signature(**inputs)

    
    # Obtener las puntuaciones
    logits_per_image = outputs.logits_per_image  # Las puntuaciones para cada imagen
    probs = logits_per_image.softmax(dim=1)  # Convertir las puntuaciones a probabilidades

    # Obtener el índice de la clase con la mayor probabilidad
    predicted_class_idx = probs.argmax(dim=1).item()

    # Obtener el nombre de la clase predicha
    predicted_class = texts[predicted_class_idx]

    if predicted_class == "signature":
        return True
    return False


"""
Función para recortar las imágenes de los caracteres individuales 
"""
def cropCharacters(image):
    #print(image.size)
    imageArray = np.array(image)
    gray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral para obtener una imagen binaria
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcular el área total de la imagen
    total_area = imageArray.shape[0] * imageArray.shape[1]

    # Calcular un umbral mínimo en función del área total de la imagen
    min_area_threshold_ratio = 0.001  # Ajusta este valor según sea necesario
    min_area_threshold = total_area * min_area_threshold_ratio

    bounding_boxes = []
    # Iterar sobre los contornos y dibujar rectángulos alrededor de los contornos relevantes
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageArray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bounding_boxes.append((x, y, x+w, y+h))
    

    # Ordenar las bounding boxes por coordenada x
    bounding_boxes.sort(key=lambda x: x[0])

    # Recorte de la imagen original usando las bounding boxes
    characters_pil = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        character = image.crop((x1-1, y1-1, x2+1, y2+1))
        characters_pil.append(character)

    return characters_pil

"""
Función para obtener la letra de una imagen mediante la red CNN
"""
def getLetter(image):

    image = preprocess_CNN(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = cnn_letter(image)
        _, predicted = torch.max(output, 1)
    
    # Correspondencia entre el número predicho y el carácter
    predicted_decoded = letters_dict[predicted.item()]
    return predicted_decoded

"""
Función para obtener el dígito de una imagen mediante la red CNN
"""
def getDigit(image):
    image = preprocess_CNN(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = cnn_number(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

"""
Función encargada de extraer el DNI de una imagen, devolviendo el texto extraído.
"""
def extractDNI(image):
    image = image.crop((8,15, image.width-4, image.height-15))
    cropped_characters = cropCharacters(image)
    generated_text = ""
    if len(cropped_characters) == 9 or len(cropped_characters) >= 10:
        for i in range(0, 8):
            generated_text += str(getDigit(cropped_characters[i]))
        generated_text += getLetter(cropped_characters[8])
    else:
        for i in range(0, len(cropped_characters)):
            generated_text += str(getDigit(cropped_characters[i]))
    return generated_text

            