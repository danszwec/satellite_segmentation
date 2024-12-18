
import os
import random
import shutil
import numpy as np
from torch.utils.data import DataLoader
from utils.define_datasetclass import SegmentationDataset
import utils.image_utils
from datetime import datetime
from torchvision import transforms as T  
from PIL import Image


date = str(datetime.now().strftime("%Y-%m-%d"))

def train_dir(model_name):
    """
    Description: Create a directory to save the training results for a given model.

    Args: model_name (str): Name of the model.

    Returns: None
    """
    train_name = model_name+date
    save_dir = os.path.join('results',train_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,"checkpoints"), exist_ok=True)
    return save_dir

def get_directories(base_dir: str ,test_mode=None):
    """
    Construct and return the paths to directories for training and validation images and annotations.

    Args:
        base_dir (str): Base directory where the dataset is located.

    Returns:
        tuple: A tuple containing four paths:
            - Path to the training images directory.
            - Path to the training annotations directory.
            - Path to the validation images directory.
            - Path to the validation annotations directory.
    """
    if test_mode:
        image_train_dir = os.path.join(base_dir, 'images', 'test')
        mask_train_dir = os.path.join(base_dir, 'annotations', 'test')
        image_val_dir = os.path.join(base_dir, 'images', 'test')
        mask_val_dir = os.path.join(base_dir, 'annotations', 'test')
        return image_train_dir, mask_train_dir, image_val_dir, mask_val_dir
        

    image_train_dir = os.path.join(base_dir, 'images', 'training')
    mask_train_dir = os.path.join(base_dir, 'annotations', 'training')
    image_val_dir = os.path.join(base_dir, 'images', 'validation')
    mask_val_dir = os.path.join(base_dir, 'annotations', 'validation')
    return image_train_dir, mask_train_dir, image_val_dir, mask_val_dir

def select_transform(config,test_mode = None):
    """
    Select and return image transformations for training and validation based on the given configuration.

    Args:
        config (dict): Configuration dictionary specifying which transformations to apply.
        test_mode (bool, optional): If True, return an empty list of transformations for validation. Defaults to None.

    Returns:
        tuple: A tuple containing two lists:
            - List of transformations to apply during training.
            - List of transformations to apply during validation (usually empty if test_mode is True).
    """
    if test_mode:
        train_transform = []
        return train_transform
    my_dict = { 'Rotation' :  T.RandomRotation(1),
    'horizontal_flip' :T.RandomHorizontalFlip(p=1), 'vertical_flip': T.RandomVerticalFlip(p=1),
    'ColorJitter':T.ColorJitter(brightness=random.uniform(0.1, 0.3), contrast=random.uniform(0.1, 0.3), saturation=random.uniform(0.1, 0.3), hue=random.uniform(0, 0.1))}
    
    train_transform = [my_dict[key] for key in my_dict if config['transformes']['types'][key] is True]
    
    return train_transform

def load_data(cfg,desirable_class,batch_size,data_dir,test_mode = None):
    #Select the Augmentation
    train_transform  = select_transform(cfg,test_mode)

    #get the data directories
    image_train_dir, mask_train_dir, image_val_dir, mask_val_dir = get_directories(data_dir,test_mode)

    train_dataset = SegmentationDataset(
        image_dir=image_train_dir,
        mask_dir=mask_train_dir, 
        transform = train_transform,
        number_class = desirable_class)

    val_dataset = SegmentationDataset(
        image_dir=image_val_dir, 
        mask_dir=mask_val_dir, 
        transform = [],
        number_class = desirable_class)

    #DataLoader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4,drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4,drop_last=False)

    return train_loader, val_loader



####image handeling#######

def split_image(image_path, target_size=(512, 512)):
    """
    Splits an image into smaller patches of the specified size, pads smaller patches with black pixels,
    and saves them in a temporary directory.

    Args:
    - image_path (str): Path to the input image.
    - target_size (tuple): Target size of the patches (width, height).

    Returns:
    - str: Path to the temporary directory containing the patches.
    - tuple: Original size of the input image (width, height).
    """
    # Create a temporary directory in the same location as the image
    image_dir = os.path.dirname(image_path)
    temp_dir = os.path.join(image_dir, "temp_patches")
    os.makedirs(temp_dir, exist_ok=True)

    # Open the image
    image = Image.open(image_path)
    if image.mode != "RGB":
        print(f"Warning: Image {image_path} is not RGB. Converting...")
        image = image.convert("RGB")

    # Image dimensions
    width, height = image.size

    # Split the image into patches with padding
    for y in range(0, height, target_size[1]):
        for x in range(0, width, target_size[0]):
            # Create a blank black image for padding
            patch = Image.new("RGB", target_size, color=(0, 0, 0))

            # Crop part of the original image
            box = (x, y, min(x + target_size[0], width), min(y + target_size[1], height))
            cropped = image.crop(box)

            # Paste the cropped region onto the black patch
            patch.paste(cropped, (0, 0))

            # Save patch
            patch_filename = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{x}_{y}.png")
            patch.save(patch_filename)

    return temp_dir, (width, height)

def rebuild_image(temp_dir, original_size, image_path, target_size=(512, 512)):
    """
    Rebuilds the original image from patches by removing black padding and deletes the temporary directory.

    Args:
    - temp_dir (str): Path to the temporary directory containing the patches.
    - original_size (tuple): The original size of the image (width, height).
    - image_path (str): Path to the original image.
    - target_size (tuple): Target size of the patches (width, height).

    Returns:
    - str: Path to the rebuilt image.
    """
    # Create a new blank image with the original size
    rebuilt_image = Image.new("RGB", original_size)

    # Read patches and paste them into the blank image (removing padding)
    for patch_name in sorted(os.listdir(temp_dir)):
        patch_path = os.path.join(temp_dir, patch_name)
        patch = Image.open(patch_path)

        # Extract coordinates from the filename
        _, x, y = os.path.splitext(patch_name)[0].rsplit("_", 2)
        x, y = int(x), int(y)

        # Determine the actual region to paste (crop black padding if necessary)
        box = (0, 0, min(target_size[0], original_size[0] - x), min(target_size[1], original_size[1] - y))
        cropped_patch = patch.crop(box)

        # Paste the cropped patch into the correct position
        rebuilt_image.paste(cropped_patch, (x, y))

    # Save the rebuilt image
    rebuilt_image_path = os.path.join(
        os.path.dirname(image_path), f"{os.path.splitext(os.path.basename(image_path))[0]}_rebuilt.png"
    )
    rebuilt_image.save(rebuilt_image_path, "PNG")

    # Cleanup: Delete the temporary directory and its contents
    shutil.rmtree(temp_dir)

    return rebuilt_image_path


######divide data#####################

def calculate_one_percent(directory,train_percents=0.8,val_percents=0.1):
    try:
        # Count the number of files
        num_files = len(os.listdir(directory))
        # Calculate percents
        train_num = int(num_files * train_percents)
        val_num = int(num_files * val_percents)
        test_num = num_files - train_num - val_num
        return [train_num, val_num, test_num]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def divide_images(images_path,new_path,train_percents=0.8,val_percents=0.1):
    try:
        # Get the list of files
        files = os.listdir(images_path)

        # Calculate the number of files
        sizes = calculate_one_percent(images_path,train_percents,val_percents)

        #make 3 dir for train, val, test
        os.makedirs(os.path.join(new_path, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'train', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'val', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(new_path, 'test', 'masks'), exist_ok=True)

        # Divide the data randomaly into train, val, and test and remove them to the corresponding folders
        for i in range(sizes[0]):
            file = random.choice(files)
            os.rename(os.path.join(images_path, file), os.path.join(new_path, 'train', 'images', file))
            files.remove(file)
        for i in range(sizes[1]):
            file = random.choice(files)
            os.rename(os.path.join(images_path, file), os.path.join(new_path, 'val', 'images', file))
            files.remove(file)
        for file in files:
            os.rename(os.path.join(images_path, file), os.path.join(new_path, 'test', 'images', file))
        
        print("The images has been divided successfully")

    except Exception as e:
        print(f"An error occurred: {e}")

def divide_masks(masks_path,images_path,new_path):
    try:

        #dirs for images
        train_img_path = os.path.join(new_path, 'train', 'images')
        val_img_path = os.path.join(new_path, 'val', 'images')
        test_img_path = os.path.join(new_path, 'test', 'images')

        # Get the list of files
        masks_files = os.listdir(masks_path)
        train_files = os.listdir(train_img_path)
        val_files = os.listdir(val_img_path)
        test_files = os.listdir(test_img_path)

        # Get the common files between the masks and images
        train_files = list(set(masks_files) & set(train_files))
        val_files = list(set(masks_files) & set(val_files))
        test_files = list(set(masks_files) & set(test_files))

        #move the common files to the train and val dirs
        for file in train_files:
            os.rename(os.path.join(masks_path, file), os.path.join(new_path, 'train', 'masks', file))
            masks_files.remove(file)
        for file in val_files:
            os.rename(os.path.join(masks_path, file), os.path.join(new_path, 'val', 'masks', file))
            masks_files.remove(file)
        for file in test_files:
            os.rename(os.path.join(masks_path, file), os.path.join(new_path, 'test', 'masks', file))
            masks_files.remove(file)
        print("The masks has been divided successfully")

    except Exception as e:
        print(f"An error occurred: {e}")

    return

def divide_data(data_path = None,new_path = None):
    images_path = os.path.join(data_path, 'images')
    masks_path = os.path.join(data_path, 'masks')
    divide_images(images_path,new_path)
    divide_masks(masks_path,images_path,new_path)



#### cvat convertor ####
def parse_color_file(base_dir):
    txt_path = os.path.join(base_dir, 'labelmap.txt')
    label_dict = {}
    with open(txt_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            if line.strip() :  # Ignore empty lines
                label, color, *_ = line.split(':')
                color_rgb = np.array([*map(int, color.split(','))])
                label_dict[label] = color_rgb
    return label_dict

def fetch_masks(base_dir):
    # where the masks are located
    masks_path = os.path.join(base_dir, 'SegmentationClass')

    # create a directory to save the annotations
    annotations_dir = os.path.join(date ,'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    # Get the list of files
    masks_files = os.listdir(masks_path)

    # Get the color dictionary
    label_dict =parse_color_file(base_dir)

    # Convert the masks to grayscale and save them
    for file in masks_files:
        mask = np.array(Image.open(os.path.join(masks_path, file)))
        grey_mask = Image.fromarray(utils.image_utils.rgb_to_grey(mask, label_dict))
        grey_mask.save(os.path.join(annotations_dir, file))
    return masks_files

def fetch_images(base_dir,masks_files):
    # where the images are located
    images_path = os.path.join(base_dir, 'JPEGImages')

    # create a directory to save the images
    images_dir = os.path.join(date ,'images')
    os.makedirs(images_dir, exist_ok=True)

    # Move the images to the new directory
    for file in masks_files:
        file_path = os.path.join(images_path, file)
        
        #open the image and save it in the new directory
        image = Image.open(file_path)
        image.save(os.path.join(images_dir, file))
    return

def pull_data(base_dir):
    # Fetch the masks and images
    masks_files = fetch_masks(base_dir)
    fetch_images(base_dir, masks_files)
    
    # Print a success message
    print("Data has been pulled successfully")
    return
