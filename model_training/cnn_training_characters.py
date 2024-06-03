import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import os
from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms


"""
Ruta del dataset de entrenamiento
"""
datasetPathRoot = "datasetSave12_junto"

"""
Ruta del directorio donde se va a guardar el modelo entrenado
"""
modelResultPathRoot = "models\\cnn_ocr_Characters_.pth"


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

class DNIDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.dirPaths = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        #image = cv2.imread(self.dirPaths[idx], cv2.IMREAD_GRAYSCALE)
        # heigth = image.size[1]
        # image = ImageOps.pad(image, (90, heigth), color='white')
        # # image.show()
        # # input("press")
        # image = ImageOps.pad(image, (95, 20), color='white')

        image_np = np.array(image.convert('RGB'))

        image_np = cv2.resize(image_np, (28, 28))

        # # # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # # # Aplicar umbral para obtener una imagen binaria
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        image = Image.fromarray(binary)

        # image.show()
        # input("press")

        if self.transform:
            image = self.transform(image)

        # print("label: ", label)
        label_string = str(label)
        etiqueta_numerica = -1
        if label_string.isnumeric():
            etiqueta_numerica = int(label)
        else:
            etiqueta_numerica = abecedario_numerado[label]
        # print(etiqueta_numerica)
        return image, torch.tensor(etiqueta_numerica, dtype=torch.long)



torch.manual_seed(42)
BATCH = 256


device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__ (self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.globalp = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Sequential(nn.Flatten(), nn.Linear(16, len(datasets.EMNIST.classes)))
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x =self.conv2(x)
        x = self.relu(x)
        print("Tamaño de entrada antes de la capa de pooling:", x.size())
        x = self.pool(x)
        print("Tamaño de salida después de la capa de pooling:", x.size())
        x = self.globalp(x)
        x = self.classify(x)
        return x

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)

# Cargar el dataset utilizando la estructura de directorios
dataset = load_dataset("imagefolder", data_dir=datasetPathRoot)

# Definir las transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = DNIDataset(dataset["train"]["image"],dataset["train"]["label"], transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#train test loop
epochs = 20
for epoch in range (epochs):
    for batch,(X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred =model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}  Training Loss {loss}  ')


torch.save(model, modelResultPathRoot)