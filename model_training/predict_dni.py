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


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predecir_dni(model, imagen_ruta):
    imagen = cv2.imread(imagen_ruta, cv2.IMREAD_GRAYSCALE)
    imagen = cv2.resize(imagen, (128, 128))
    imagen = Image.fromarray(imagen)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    imagen = transform(imagen).unsqueeze(0)
    with torch.no_grad():
        outputs = model(imagen)
        print(outputs)
        _, prediccion = torch.max(outputs.data, 1)
        digitos = prediccion[:8].numpy()
        letra = chr(prediccion[-1].item() + ord('A'))
        return ''.join(map(str, digitos)) + letra


model = CNNModel()
model.load_state_dict(torch.load('cnn_ocr_450.pth'))


# Ejemplo de uso
dni_texto = predecir_dni(model, "datasetDNI\\00403747M.png")
print(f"DNI detectado: {dni_texto}")
