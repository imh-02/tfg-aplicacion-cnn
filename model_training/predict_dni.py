import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
from datasets import load_dataset
import torch.nn.functional as F

"""
Ruta del dataset de entrenamiento
"""
datasetPathRoot = "datasetDNI"

"""
Ruta del directorio donde se va a guardar el modelo entrenado
"""
modelResultPathRoot = "cnn_ocr_1000.pth"

"""
Abecedario numerdado del a partir del 10 para A-Z
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 31 * 3, 256)
        self.fc2 = nn.Linear(256, 9*36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 31 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 9, 36)  # Reshape a [batch_size, 9, 36]
        return x

model = CNNModel().to(device)

torch.save(model.state_dict(), modelResultPathRoot)

# Cargar los pesos entrenados
model.load_state_dict(torch.load(modelResultPathRoot))

# Configurar el modelo en modo evaluación
model.eval()

# Mover el modelo a la GPU si está disponible
model = model.to(device)

# Transformación utilizada durante el entrenamiento
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Función para procesar la imagen
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (252, 28))
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir una dimensión para el batch
    return image

# Invertir el diccionario de abecedario numerado para convertir números a letras
numerado_abecedario = {v: k for k, v in abecedario_numerado.items()}

# Función para convertir etiquetas numéricas a caracteres
def labels_to_string(labels):
    label_str = ''.join(str(l.item()) for l in labels[:-1])
    label_str += numerado_abecedario[labels[-1].item()]
    return label_str

# Ruta de la imagen a predecir
image_path = 'ejemplo2.png'

# Procesar la imagen
input_image = preprocess_image(image_path)

# Mover la imagen a la GPU si está disponible
input_image = input_image.to(device)

# Realizar la predicción
with torch.no_grad():
    output = model(input_image)

print(output)

#print('Predicción:', labels_to_string(output[0].argmax(1)))
