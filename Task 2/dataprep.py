import cv2
import os
from albumentations import (
    Compose, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightnessContrast, RandomGamma, Rotate
)
import random
from sklearn.model_selection import train_test_split

IMAGES_DIR = r'C:\Users\yevge\Documents\YEVGEN\GitHub\Deep Learning\DL\Task 2\images'
OUTPUT_DIR = r'C:\Users\yevge\Documents\YEVGEN\GitHub\Deep Learning\DL\Task 2\augmented'
PATCH_SIZE = 128
QUANTITY = 400
TEST_SIZE = 0.2


image_list = []

for name in os.listdir(IMAGES_DIR):
    
    # loading image
    image = cv2.imread(IMAGES_DIR + "/" + name)
    # transforming to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_list.append(gray_image)


pipeline = Compose([
    RandomCrop(PATCH_SIZE, PATCH_SIZE),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    RandomGamma(p=0.5),
    Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5)
])


patches = []

for i in range(QUANTITY):
    index = random.randint(0,len(image_list)-1)

    transformed_data = pipeline(image=image_list[index])
    transformed_image = transformed_data['image']


    patches.append(transformed_image)


train_patches, test_patches = train_test_split(patches, test_size=TEST_SIZE)

train_dir = os.path.join(OUTPUT_DIR, "train")
test_dir = os.path.join(OUTPUT_DIR, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for i in range(len(train_patches)):
    cv2.imwrite(os.path.join(train_dir , str(i) + ".png"), train_patches[i])
for i in range(len(test_patches)):
    cv2.imwrite(os.path.join(test_dir , str(i) + ".png"), test_patches[i])








    
    
    
