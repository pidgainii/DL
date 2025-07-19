# test_autoencoder.py

from model import Autoencoder
import torch
import torch.nn as nn
from dataloader import CustomDataset
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# ------------- CONFIG -------------
TEST_DIRECTORY = r'path'
MODEL_PATH = "autoencoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------

# Cargar modelo
autoencoder = Autoencoder().to(DEVICE)
autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
autoencoder.eval()

# Dataset
test_dataset = CustomDataset(TEST_DIRECTORY)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

test_loss_function = nn.MSELoss()

for (image, _) in test_loader:
    image = image.to(DEVICE)
    reconstructed = autoencoder(image)
    loss = test_loss_function(reconstructed, image)
    print(f"Loss: {loss.item():.6f}")


indices = random.sample(range(len(test_dataset)), 5)
images = torch.stack([test_dataset[i][0] for i in indices]).to(DEVICE)

with torch.no_grad():
    reconstructions = autoencoder(images).cpu()

images = images.cpu()


pairs = []
for original, reconstructed in zip(images, reconstructions):
    orig_pil = TF.to_pil_image(original)
    recon_pil = TF.to_pil_image(reconstructed)
    pair = Image.new("L", (orig_pil.width * 2, orig_pil.height))
    pair.paste(orig_pil, (0, 0))
    pair.paste(recon_pil, (orig_pil.width, 0))
    pairs.append(pair)


comparison_img = Image.new("L", (pairs[0].width, pairs[0].height * len(pairs)))
for i, pair in enumerate(pairs):
    comparison_img.paste(pair, (0, i * pair.height))

plt.figure(figsize=(6, 12))
plt.axis("off")
plt.title("Original vs Reconstruido")
plt.imshow(comparison_img, cmap='gray')
plt.show()
