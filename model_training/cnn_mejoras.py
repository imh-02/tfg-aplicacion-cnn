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

# Rutas
datasetPathRoot = "datasets_generated\\datasetDNI95x20_40000"
modelResultPathRoot = "models\\cnn_ocr_40.000_95x20Mejoras.pth"

# Abecedario numerado
abecedario_numerado = {
    "A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15, "G": 16,
    "H": 17, "I": 18, "J": 19, "K": 20, "L": 21, "M": 22, "N": 23,
    "O": 24, "P": 25, "Q": 26, "R": 27, "S": 28, "T": 29, "U": 30,
    "V": 31, "W": 32, "X": 33, "Y": 34, "Z": 35
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DNIDataset(Dataset):
    def __init__(self, ds, root, transform=None):
        self.ds = ds
        self.root = root
        self.transform = transform
        self.dirPaths = [os.path.join(root, i) for i in os.listdir(root)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        fileName = os.path.basename(self.dirPaths[idx]).replace(".png", "")
        label = fileName
        image = Image.open(self.dirPaths[idx])
        image_np = np.array(image.convert('RGB'))
        image_np = cv2.resize(image_np, (95, 20))
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        image = Image.fromarray(binary)

        if self.transform:
            image = self.transform(image)

        etiqueta_numerica = [int(digito) for digito in label[:8]] + [abecedario_numerado[label[-1]]]
        return image, torch.tensor(etiqueta_numerica, dtype=torch.long)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 11 * 2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 9 * 36)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 11 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 9, 36)
        return x

# Cargar el dataset
dataset = load_dataset("imagefolder", data_dir=datasetPathRoot)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = DNIDataset(dataset['train'], datasetPathRoot, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = CNNModel().to(device)

print('Start Training')

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

num_classes = 36

for epoch in range(30):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.view(-1)
        labels_clipped = torch.clamp(labels, 0, num_classes - 1).long()
        outputs = outputs.view(-1, num_classes)
        loss = criterion(outputs, labels_clipped)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    scheduler.step(running_loss)

print('Finished Training')
torch.save(model, modelResultPathRoot)
