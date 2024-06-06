from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
from datasets import load_dataset 


"""
Ruta del dataset de entrenamiento
"""
datasetPathRoot = "datasetSave12\\letters"

class DigitsDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        label = self.labels[idx]

        # heigth = image.size[1]
        # width = image.size[0]

        # image = ImageOps.pad(image, (width,heigth + 4), color='white')
        # image = ImageOps.pad(image, (width + 4,heigth + 4), color='white')
        # image = ImageOps.pad(image, (128,128), color='white')
        image = image.point(lambda p: p > 200 and 255)
        image_np = np.array(image)
        image_np = cv2.bitwise_not(image_np)
        image = Image.fromarray(image_np)

        image = image.filter(ImageFilter.GaussianBlur(radius=1))

        # donwsample image 28 x 28 using bicubic
        # image = image.resize((28,28), Image.BICUBIC)
        image = image.convert('L')

        image_preprocessed = image
        transform = transforms.Compose([transforms.ToTensor()])
        image_preprocessed = transform(image)

        etiqueta_numerica = int(label+1)
        # print(etiqueta_numerica)
        return image_preprocessed, torch.tensor(etiqueta_numerica, dtype=torch.long)

dataset = load_dataset("imagefolder", data_dir=datasetPathRoot)

images_train = dataset['train']['image']
labels_train = dataset['train']['label']


train_my_data = DigitsDataset(images_train, labels_train)
train_loader = DataLoader(train_my_data, batch_size=32, shuffle=True)

class CNN_letters(nn.Module):
    def __init__(self):
        super(CNN_letters, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 27)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN_letters().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.01)

loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0}%)]\t{loss.item():.6f}')

    
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0

#     with torch.no_grad():
#         for data,target in loaders['test']:
#             data, target = data.to(device), target.to(device)
#             outputs = model(data)
#             test_loss += loss_fn(outputs, target).item()
#             pred = outputs.argmax(dim=1, keepdim = True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(loaders['test'].dataset)
#     print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n')

for epoch in range(1, 21):
    train(epoch)
    # test()

torch.save(model, "cnn_miDataset_tutorial_letters_20epochs.pth")