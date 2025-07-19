# train_autoencoder.py

from model import Autoencoder
import torch
import torch.nn as nn
from dataloader import CustomDataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
import torchvision.utils as vutils
from pytorch_msssim import ssim
import os

# ---------------- CONFIG ----------------
TRAIN_DIRECTORY = r'path'
TEST_DIRECTORY = r'path'
MODEL_PATH = "autoencoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
# ----------------------------------------

def show_random_batch(dataset, batch_size=32):
    indices = random.sample(range(len(dataset)), batch_size)
    images = [dataset[i][0].unsqueeze(0) for i in indices]
    batch = torch.cat(images, dim=0)
    grid = vutils.make_grid(batch, nrow=8, normalize=True, pad_value=1)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Random Training Batch")
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.show()

def data_loaders():
    train_data = CustomDataset(TRAIN_DIRECTORY)
    test_data = CustomDataset(TEST_DIRECTORY)
    show_random_batch(train_data)
    return (
        torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE),
        torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    )

def ssim_loss(output, target):
    return 1 - ssim(output, target, data_range=1.0, size_average=True)

# --------- Training ---------
train_loader, test_loader = data_loaders()

autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
loss_function = ssim_loss

autoencoder.train()
for epoch in range(EPOCHS):
    for (image, _) in train_loader:
        image = image.to(DEVICE)
        reconstructed = autoencoder(image)
        loss = loss_function(reconstructed, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

torch.save(autoencoder.state_dict(), MODEL_PATH)
