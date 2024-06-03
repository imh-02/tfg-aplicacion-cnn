import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
from torchvision import datasets
import torch.nn.functional as F

"""
Ruta del directorio donde se va a guardar el modelo entrenado
"""
modelResultPathRoot = "models\\cnn_rnn_model.pth"

# Definir transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Descargar y cargar el dataset EMNIST
emnist_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=False, transform=transform)
emnist_loader = DataLoader(emnist_dataset, batch_size=64, shuffle=True)

# Función para crear cadenas concatenadas de caracteres
def create_concatenated_images(dataset, num_chars=5):
    images = []
    labels = []
    for _ in range(len(dataset) // num_chars):
        chars = [dataset[i] for i in np.random.randint(0, len(dataset), num_chars)]
        concat_image = torch.cat([c[0] for c in chars], dim=2)
        concat_label = torch.tensor([c[1] for c in chars])
        images.append(concat_image)
        labels.append(concat_label)
    return images, labels


# Crear imágenes concatenadas
concat_images, concat_labels = create_concatenated_images(emnist_dataset)
concat_dataset = list(zip(concat_images, concat_labels))
concat_loader = DataLoader(concat_dataset, batch_size=16, shuffle=True)

class CNN_RNN_Model(nn.Module):
    def __init__(self, num_classes=62, hidden_dim=128, num_layers=2):
        super(CNN_RNN_Model, self).__init__()
       
        # Definir la CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
       
        # Definir la RNN (LSTM)
        self.rnn = nn.LSTM(64 * 7, hidden_dim, num_layers, batch_first=True, bidirectional=True)
       
        # Clasificador final
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
   
    def forward(self, x):
        batch_size = x.size(0)
       
        # Pasar por la CNN
        cnn_out = self.cnn(x)
       
        # Preparar la salida de la CNN para la RNN
        cnn_out = cnn_out.permute(0, 2, 1, 3)
        cnn_out = cnn_out.view(batch_size, cnn_out.size(1), -1)
       
        # Pasar por la RNN
        rnn_out, _ = self.rnn(cnn_out)
       
        # Pasar por el clasificador
        rnn_out = self.fc(rnn_out)
       
        return rnn_out

num_classes = 62  # Número de clases en EMNIST (10 dígitos + 26 letras mayúsculas + 26 letras minúsculas)
learning_rate = 0.001
num_epochs = 10

# Inicializar el modelo, la pérdida y el optimizador
model = CNN_RNN_Model(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Función de entrenamiento
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # Pasar los datos por el modelo
            outputs = model(images)
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
           
            # Calcular la pérdida
            loss = criterion(outputs, labels)
           
            # Retropropagación y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Entrenar el modelo
train_model(model, concat_loader, criterion, optimizer, num_epochs)

torch.save(model, modelResultPathRoot)

# Evaluación del modelo
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in concat_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 2)
        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')