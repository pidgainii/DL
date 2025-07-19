import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=2, in_channels=1, out_channels=16, padding=1),
            nn.ReLU(True),
            nn.Conv2d(kernel_size=3, stride=2, in_channels=16, out_channels=32, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32), # number of channels is 32
            nn.Conv2d(kernel_size=3, stride=2, in_channels=32, out_channels=64, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features=16*16*64, out_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=16*16*64),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(kernel_size=3, stride=2, in_channels=64, out_channels=32, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(kernel_size=3, stride=2, in_channels=32, out_channels=16, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(kernel_size=3, stride=2, in_channels=16,  out_channels=1, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    