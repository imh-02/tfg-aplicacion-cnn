import torch
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from routes import table_model_path, device

if device == True:
    device = 'cuda'
else:
    device = 'cpu'
print("Using device:", device)

"""
Carga de los modelos preentrenados para la extractracción de las filas y columnas de una tabla, desde el almacenamiento local.
"""
model = TableTransformerForObjectDetection.from_pretrained(table_model_path).to(device)
feature_extractor = DetrImageProcessor.from_pretrained(table_model_path)

"""
Función encargada de extraer todas las filas y columnas, mediante un modelo pre entrenado de Transformer
"""
def cell_detection(image):
    width, height = image.size
    image.resize((int(width*0.5), int(height*0.5)))


    encoding = feature_extractor(image, return_tensors="pt").to(device)
    encoding.keys()

    with torch.no_grad():
      outputs = model(**encoding)


    target_sizes = [image.size[::-1]]
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.85, target_sizes=target_sizes)[0]

    return results