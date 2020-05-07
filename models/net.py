import math
import torch
import torch.nn as nn
import torch.nn.intrinsic as inn
import torch.quantization as quant
from functools import reduce


class Net(nn.Module):
    input_shape = [1, 3, 32, 32]
    kernel_size = 3
    default_arch = ((kernel_size, 10, 1), (kernel_size, 15, 2), (kernel_size, 20, 1))

    def __init__(self, conv_layers=default_arch, num_classes=2):
        super(Net, self).__init__()
        # input 3x48x48
        if conv_layers == None:
            kern_size = 5
            conv_layers = ((kern_size, 6), (kern_size, 16))
        self.quant = quant.QuantStub()

        self.features = self.init_features(conv_layers)
        self.fc = self.init_fc(num_classes)

        self.dequant = quant.DeQuantStub()

    def init_features(self, conv_layers_params):
        padding = 0
        stride = 1
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        convs = []
        prev_nkernels = 3
        for conv_layer in conv_layers_params:
            kern_size = conv_layer[0]
            nkernels = conv_layer[1]
            stride = conv_layer[2]
            if stride > 1:
                padding = kern_size // 2
            else:
                padding = 0
            convs.append(inn.ConvBnReLU2d(nn.Conv2d(prev_nkernels, nkernels, kern_size,
                                                    padding=padding),
                                          nn.BatchNorm2d(nkernels),
                                          nn.ReLU()))
            convs.append(self.dropout)
            prev_nkernels = nkernels
        convs.append(self.pool)
        features = nn.Sequential(*convs)
        test_inp = torch.rand(self.input_shape)
        features_shape = features(test_inp).shape
        self.flat_features_len = reduce(lambda x, y: x * y, [1, *features_shape[1:]])
        return features

    def init_fc(self, num_classes):
        self.fc1 = inn.LinearReLU(nn.Linear(self.flat_features_len, 120), nn.ReLU())
        self.fc2 = inn.LinearReLU(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)
        return nn.Sequential(self.fc1, self.fc2, self.fc3)

    def update_fc(self, num_classes):
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.reshape(-1, self.flat_features_len)
        x = self.fc(x)
        x = self.dequant(x)
        return x
