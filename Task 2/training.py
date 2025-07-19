from model import Autoencoder
import torch
import torch.nn as nn
from dataloader import CustomDataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
import torchvision.utils as vutils
from pytorch_msssim import ssim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os




TRAIN_DIRECTORY = r'C:\Users\yevge\Documents\YEVGEN\GitHub\Deep Learning\DL\Task 2\augmented\train'
TEST_DIRECTORY = r'C:\Users\yevge\Documents\YEVGEN\GitHub\Deep Learning\DL\Task 2\augmented\test'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 50



# a function to sample a random training batch from the list of training tiles
def show_random_batch(dataset, batch_size=32):
    indices = random.sample(range(len(dataset)), batch_size)
    images = [dataset[i][0].unsqueeze(0) for i in indices]  # get image only
    batch = torch.cat(images, dim=0)
    grid = vutils.make_grid(batch, nrow=8, normalize=True, pad_value=1)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Random Training Batch")
    plt.imshow(grid.permute(1, 2, 0).cpu())  # CHW -> HWC for matplotlib
    plt.show()




def data_loaders():
    train_data = CustomDataset(TRAIN_DIRECTORY)
    test_data = CustomDataset(TEST_DIRECTORY)


    # plot random batch
    show_random_batch(train_data)

    return torch.utils.data.DataLoader(train_data, batch_size=32), torch.utils.data.DataLoader(test_data, batch_size=32)



# we need a data loader for our model
train_loader, test_loader = data_loaders()



def ssim_loss(output, target):
    # SSIM devuelve similaridad -> 1 es perfecto
    # Por eso lo convertimos en una "pérdida": 1 - ssim
    return 1 - ssim(output, target, data_range=1.0, size_average=True)







# 1
autoencoder = Autoencoder().to(DEVICE)
autoencoder.train(True)
# 2
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
loss_function = ssim_loss
outputs = []





# regardless the batch size, in one epoch all images are processed
# batch size 32 means that 32 images are processed at the same time
for epoch in range(epochs):
    for (image, _) in train_loader:

        reconstructed = autoencoder(image)
        loss = loss_function(reconstructed, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    
    outputs.append((epoch,image,reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")


# # state dict saves the parameters of the already trained model
# torch.save(autoencoder.state_dict(), "autoencoder.pth")




# ------------------------- TESTING -------------------------
print("\nTesting...\n")

autoencoder.eval()

test_loss_function = nn.MSELoss()


recon_images = []
for (image, _) in test_loader:

    reconstructed = autoencoder(image)
    loss = test_loss_function(reconstructed, image)
    recon_images.append(reconstructed)

    print(f"Loss: {loss.item():.6f}")








print("\nVisualizing 5 original vs reconstructed images...\n")

# Sample 5 random test images
test_dataset = test_loader.dataset
indices = random.sample(range(len(test_dataset)), 5)
images = torch.stack([test_dataset[i][0] for i in indices]).to(DEVICE)

# Reconstruct with autoencoder
with torch.no_grad():
    reconstructions = autoencoder(images).cpu()

images = images.cpu()

# Create side-by-side comparison image pairs
pairs = []
for original, reconstructed in zip(images, reconstructions):
    # Convert each to PIL image for easier side-by-side plotting
    orig_pil = TF.to_pil_image(original)
    recon_pil = TF.to_pil_image(reconstructed)

    # Concatenate images horizontally
    pair = Image.new("L", (orig_pil.width * 2, orig_pil.height))
    pair.paste(orig_pil, (0, 0))
    pair.paste(recon_pil, (orig_pil.width, 0))
    pairs.append(pair)

# Combine all pairs vertically
comparison_img = Image.new("L", (pairs[0].width, pairs[0].height * len(pairs)))
for i, pair in enumerate(pairs):
    comparison_img.paste(pair, (0, i * pair.height))

# Show final comparison image
plt.figure(figsize=(6, 12))
plt.axis("off")
plt.title("Original vs Reconstructed (5 Random Examples)")
plt.imshow(comparison_img, cmap='gray')
plt.show()


