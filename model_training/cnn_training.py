import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
from datasets import load_dataset
import torch.nn.functional as F
from sklearn.model_selection import KFold

"""
Ruta del dataset de entrenamiento
"""
datasetPathRoot = "datasets_generated\\datasetDNI95x20_10000"

"""
Ruta del directorio donde se va a guardar el modelo entrenado
"""
modelResultPathRoot = "cnn_ocr_1000_completo95x20_10000-aug.pth"

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

class DNIDataset(Dataset):
    def __init__(self, ds, root, transform=None):
        self.ds = ds
        self.root = root
        self.transform = transform
        self.dirPaths = []

        for i in os.listdir(root):
            self.dirPaths.append(root + "\\" + i)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        fileName = self.dirPaths[idx].replace(self.root + "\\", "")
        fileName = fileName.replace(".png", "")

        label = fileName


        image = cv2.imread(self.dirPaths[idx], cv2.IMREAD_GRAYSCALE)
        #image = cv2.resize(image, (95, 20))
        image = Image.fromarray(image)
        image = ImageOps.pad(image, (95, 20), method=Image.BICUBIC)

        if self.transform:
            image = self.transform(image)

        # print("label: ", label)

        etiqueta_numerica = [int(digito) for digito in label[:8]] + [abecedario_numerado[label[-1]]]
        # print(etiqueta_numerica)
        return image, torch.tensor(etiqueta_numerica, dtype=torch.long)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 11 * 2, 256)  # Ajuste del tamaño de la capa lineal
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 9 * 36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 11 * 2)  # Ajuste del tamaño de la vista
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 9, 36)
        return x



# Cargar el dataset utilizando la estructura de directorios
dataset = load_dataset("imagefolder", data_dir=datasetPathRoot)

# Definir las transformaciones
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

transform = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),  # Añadir augmentación
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = DNIDataset(dataset['train'], datasetPathRoot, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CNNModel().to(device)

print('Start Training')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Define el número total de clases
num_classes = 36

# Dentro del bucle de entrenamiento
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Mover los datos a la GPU
        optimizer.zero_grad()
        outputs = model(inputs)

        labels = labels.view(-1)
        
        # Asegúrate de que las etiquetas estén en el rango correcto
        labels_clipped = torch.clamp(labels, 0, num_classes - 1).long()

        outputs = outputs.view(-1, num_classes)
        
        loss = criterion(outputs, labels_clipped)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100}')
            running_loss = 0.0


print('Finished Training')

torch.save(model, modelResultPathRoot)









