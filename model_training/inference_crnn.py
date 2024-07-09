import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import string
from sklearn.preprocessing import LabelEncoder


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
    
transform = transforms.Compose([
    transforms.Resize((32, 128)),  # Altura fija y ancho variable
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])




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

# cargar modelo 
model = torch.load("cnn_emnistCompleto_blank_60_epochs.pth")
model.eval()

# cargar imagen
with torch.no_grad():
    image = Image.open("test3.png").convert('L')
    image = transform(image).unsqueeze(0)
    output = model(image)   
    _, prediction = output.max(2)
    prediction = prediction.transpose(1, 0).contiguous().view(-1)

    # Eliminar predicciones repetidas y tokens "blank" (no eliminar '0' válidos)
    decoded_prediction = []
    for i in range(len(prediction)):
        if i != 0 and prediction[i] == prediction[i-1]:
            continue  # Ignorar predicciones repetidas
        if prediction[i] != blank_index:  # Ignorar solo el token de "blank"
            decoded_prediction.append(prediction[i].item())

    predicted_text = decode_label(decoded_prediction)
    print(f'Predicted Text: {predicted_text}')

