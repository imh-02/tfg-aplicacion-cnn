import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import string
from sklearn.preprocessing import LabelEncoder

# images paths
root_dir = 'datasets_generated\\datasetDNI'

characters = string.digits + string.ascii_uppercase  # '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# añadir blank character _
characters = characters + '_'
blank_index = len(characters) - 1

label_encoder = LabelEncoder()
label_encoder.fit(list(characters))

def encode_label(label):
    return label_encoder.transform(list(label))

def decode_label(encoded_label):
    return ''.join(label_encoder.inverse_transform(encoded_label))

class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        label = encode_label(label)
        return image, torch.tensor(label, dtype=torch.long)
    
transform = transforms.Compose([
    transforms.Resize((32, 128)),  # Altura fija y ancho variable
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



image_paths = []
labels = []

files = os.listdir(root_dir)

for file in files:
    if file.endswith('.png'):
        image_paths.append(os.path.join(root_dir, file))
        labels.append(file.split('.png')[0])

print('Loading dataset...')

dataset = OCRDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        return output

model = CRNN(imgH=32, nc=1, nclass=len(characters), nh=256)
criterion = nn.CTCLoss(blank=blank_index, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 60

print('Start training')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        
        # Transforma outputs y labels para CTC loss
        input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
        target_lengths = torch.full(size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.long)

        loss = criterion(outputs.log_softmax(2), labels, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')


torch.save(model, "cnn_emnistCompleto_blank_60_epochs.pth")