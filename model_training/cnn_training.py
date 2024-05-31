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
modelResultPathRoot = "cnn_ocr_5000.pth"

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
        image = cv2.resize(image, (128, 128))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        etiqueta_numerica = [int(digito) for digito in label[:-1]] + [ord(label[-1]) - ord('A')]
        return image, torch.tensor(etiqueta_numerica, dtype=torch.long)


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

model = CNNModel().to(device)


# Cargar el dataset utilizando la estructura de directorios
dataset = load_dataset("imagefolder", data_dir=datasetPathRoot)

# Definir las transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = DNIDataset(dataset['train'], datasetPathRoot, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
        
        # Asegúrate de que las etiquetas estén en el rango correcto
        labels_clipped = torch.clamp(labels, 0, num_classes - 1).float()
        
        loss = criterion(outputs, labels_clipped)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100}')
            running_loss = 0.0


print('Finished Training')

torch.save(model.state_dict(), modelResultPathRoot)




