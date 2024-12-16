import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from image_utils import *
from utilz import select_transform , tensor_to_image
from utils.loss_utils import get_directories
import yaml
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, number_class=7):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        self.number_class = number_class
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])


        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply transformations
        image,transforms_list = transform_image(image,self.transform)
        mask = transform_mask(mask,transforms_list)
        
        return image, mask




if __name__ == "__main__":
    with open('/home/oury/Documents/Segmentation_Project/config.yaml', 'rt') as f:
        cfg = yaml.safe_load(f.read())
    data_dir = '/data/landcover_compress/512'
    #Pull the Dataset directories
    image_train_dir, mask_train_dir, image_val_dir, mask_val_dir = get_directories(data_dir)

    #Select the Augmentation methods
    train_transform,val_transform  = select_transform(cfg)

    train_dataset = SegmentationDataset(
        image_dir=image_train_dir,
        mask_dir=mask_train_dir, 
        transform = train_transform,
        number_class = 4)
    transform = transforms.ToTensor()
    for i in range(500):
        tensor_tuple = train_dataset.__getitem__(i)
        
        rgb_tensor = (tensor_to_image(tensor_tuple[0]).cpu().numpy().transpose(1, 2, 0))  # Move channels to last dimension
        grayscale_tensor = tensor_tuple[1]  # No need to transpose for grayscale
        grayscale_tensor = (grayscale_tensor.argmax(dim=0)).cpu().numpy()
        grayscale_tensor = coloring_masks(grayscale_tensor)
        grayscale_tensor = grayscale_tensor.cpu().numpy().transpose(1, 2, 0)
        titles=('RGB Tensor', 'Grayscale Tensor')
        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot RGB tensor
        axs[0].imshow(rgb_tensor)
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        # Plot Grayscale tensor
        axs[1].imshow(grayscale_tensor)
        axs[1].set_title(titles[1])
        axs[1].axis('off')

        plt.show()

        print("a")
