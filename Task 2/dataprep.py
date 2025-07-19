import cv2
import os
from albumentations import (
    Compose, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightnessContrast, RandomGamma, Rotate
)
import random

# Paths
BASE_DIR = r'path'
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
TRAINING_DIR = os.path.join(IMAGES_DIR, 'training')
TESTING_DIR = os.path.join(IMAGES_DIR, 'testing')
OUTPUT_DIR = os.path.join(BASE_DIR, 'augmented')

# Parameters
PATCH_SIZE = 128
TRAIN_QUANTITY = 400
TEST_QUANTITY = 100  # Ajusta este número si quieres más/menos
AUGMENTATION_PIPELINE = Compose([
    RandomCrop(PATCH_SIZE, PATCH_SIZE),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    RandomGamma(p=0.5),
    Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5)
])

def load_grayscale_images(directory):
    images = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        image = cv2.imread(path)
        if image is None:
            continue
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(gray_image)
    return images

def generate_patches(images, quantity, pipeline):
    patches = []
    for _ in range(quantity):
        index = random.randint(0, len(images) - 1)
        transformed = pipeline(image=images[index])
        patches.append(transformed['image'])
    return patches

# Load images
train_images = load_grayscale_images(TRAINING_DIR)
test_images = load_grayscale_images(TESTING_DIR)

# Augment
train_patches = generate_patches(train_images, TRAIN_QUANTITY, AUGMENTATION_PIPELINE)
test_patches = generate_patches(test_images, TEST_QUANTITY, AUGMENTATION_PIPELINE)

# Save patches
train_output_dir = os.path.join(OUTPUT_DIR, "train")
test_output_dir = os.path.join(OUTPUT_DIR, "test")
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

for i, patch in enumerate(train_patches):
    cv2.imwrite(os.path.join(train_output_dir, f"{i}.png"), patch)

for i, patch in enumerate(test_patches):
    cv2.imwrite(os.path.join(test_output_dir, f"{i}.png"), patch)
