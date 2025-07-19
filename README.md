# Deep Learning Tasks

Task 1 will not be published, as it was too simple

## Task 2: Encoder-Decoder

This programme implements a Convolutional Encoder-Decoder in Pytorch.

First we augment a small dataset of 7 faces and create a training an test set (dataprep.py file).
The augmented images are grayscale 128 x 128 pixels.

Then the model trains to encode the images into size 16 x 16 pixels, and 64 channels,
and to decode to original size.

Architecture of the model:

```
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
```

Trained model is available as autoencoder.pth ready to use


## Screenshots:

Augmented images:

![Image1](<Screenshots/Task 2/augmented.png>)

Reconstruction example:

![Image2](<Screenshots/Task 2/reconstructed.png>)

Different loss function: 

![Image3](<Screenshots/Task 2/reconstructed2.png>)

