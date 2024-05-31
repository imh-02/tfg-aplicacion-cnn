import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoModel, AutoProcessor
from routes import trocr_model_path, signature_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Carga del modelo preentrenado para la extracción de texto de una imagen, desde el almacenamiento local.
"""
processorTROCR = TrOCRProcessor.from_pretrained(trocr_model_path)
modelTROCR = VisionEncoderDecoderModel.from_pretrained(trocr_model_path).to(device)

"""
Carga del modelo preentrenado para la clasificación de imágenes de firma, desde el almacenamiento local.
"""
signature_proccessor = AutoProcessor.from_pretrained(signature_model_path)
model_signature = AutoModel.from_pretrained(signature_model_path).to(device)


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
Función encargada de extraer el DNI de una imagen, devolviendo el texto extraído.
"""
def extractDNI(image):
    image = image.crop((0,13, image.width, image.height-15))
    pixel_values = processorTROCR(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = modelTROCR.generate(pixel_values, max_new_tokens=9)
    generated_text = processorTROCR.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text