from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.transform = transforms.ToTensor()  # Convert PIL to tensor

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.filenames[idx])
        image = Image.open(img_path).convert('L')  # Grayscale
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label