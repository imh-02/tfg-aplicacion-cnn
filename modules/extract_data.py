import torch
from transformers import  AutoModel, AutoProcessor
from routes import signature_model_path, cnn_model_path
from modules.preprocess import preprocess_imageCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""
Carga del modelo con las clases de la red CNN
"""
model_cnn_ocr = torch.load(cnn_model_path).to(device)

"""
Carga del modelo preentrenado para la clasificación de imágenes de firma, desde el almacenamiento local.
"""
signature_proccessor = AutoProcessor.from_pretrained(signature_model_path)
model_signature = AutoModel.from_pretrained(signature_model_path).to(device)

"""
Abecedario numerado a partir del 10 para A-Z para CNN
"""
abecedario_numerado = {
    "A": 10,
    "B": 11,
    "C": 12,
    "D": 13,
    "E": 14,
    "F": 15,
    "G": 16,
    "H": 17,
    "I": 18,
    "J": 19,
    "K": 20,
    "L": 21,
    "M": 22,
    "N": 23,
    "O": 24,
    "P": 25,
    "Q": 26,
    "R": 27,
    "S": 28,
    "T": 29,
    "U": 30,
    "V": 31,
    "W": 32,
    "X": 33,
    "Y": 34,
    "Z": 35
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
Abecedario numerado a partir del 10 para A-Z para CNN invertido
"""
numerado_abecedario = {v: k for k, v in abecedario_numerado.items()}

"""
Función para convertir etiquetas numéricas a caracteres
"""
def labels_to_string(labels):
    label_str = ''.join(str(l.item()) for l in labels[:-1])
    label_str += numerado_abecedario[labels[-1].item()]
    return label_str


"""
Función encargada de extraer el DNI de una imagen, devolviendo el texto extraído.
"""
def extractDNI(image):
    image = image.crop((8,15, image.width-4, image.height-15))
    #
    # image = image.convert('L')
    # image.show()
    # input("recorte")
    # image.show()
    # input("press")
    input_image = preprocess_imageCNN(image)

    input_image = input_image.to(device)
    model_cnn_ocr.eval()
    with torch.no_grad():
        output = model_cnn_ocr(input_image)

    predicted = torch.argmax(output, dim=2).squeeze(0)
    generated_text = labels_to_string(predicted)
    # print('Predicción:', generated_text)
    # input("press")
    return generated_text