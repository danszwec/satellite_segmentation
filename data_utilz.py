from define_datasetclass import SegmentationDataset 
import os
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision import transforms as T  
import random

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

def get_directories(base_dir: str ,test_mode):
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

def load_data(cfg,desirable_class,batch_size,data_dir,test_mode = False):
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