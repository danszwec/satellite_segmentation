import torch
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn   
from sklearn.metrics import confusion_matrix
import os
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from image_utils import *
from colorama import Fore, Style, init
from sklearn.metrics import precision_recall_curve

# Definition Methods
def get_directories(base_dir: str):
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
    image_train_dir = os.path.join(base_dir, 'images', 'training')
    mask_train_dir = os.path.join(base_dir, 'annotations', 'training')
    image_val_dir = os.path.join(base_dir, 'images', 'validation')
    mask_val_dir = os.path.join(base_dir, 'annotations', 'validation')
    return image_train_dir, mask_train_dir, image_val_dir, mask_val_dir

def weight_tensor(train_annotations_dir , number_of_class):
    """
    Compute class weights based on pixel frequencies in the training annotations for the BCE loss function.

    Args:
        train_annotations_dir (str): Directory containing the mask images.
        number_of_class (int): Number of classes in the segmentation task.
        device (str or torch.device): The device to which the tensor should be moved. Default is 'cpu'.

    Returns:
        torch.Tensor: A tensor of shape (number_of_class, 1, 1) with the computed weights.
    """
# Directory containing your mask images
    class_counts = np.zeros(number_of_class)
    inverse_frequencies = np.zeros(number_of_class)
    for filename in os.listdir(train_annotations_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load image
            image_path = os.path.join(train_annotations_dir, filename)
            mask = Image.open(image_path).convert('L')  # Convert to grayscale
            mask = np.array(mask)
            mask = class_reduction(mask,number_of_class)
            for i in range(len(class_counts)):
                class_counts[i] += (mask==i).sum().item()
    # Get class frequencies
    for i in range(len(class_counts)):
                inverse_frequencies[i] = 1 / class_counts[i]
    # Normalize weights
    sum_inverse_frequencies = sum(inverse_frequencies)
    pos_weight = torch.tensor([weight / sum_inverse_frequencies for weight in inverse_frequencies])
    pos_weight = (pos_weight.view(number_of_class, 1, 1)).to(device)
    return pos_weight

# Select the Loss method
def select_loss(criterion_name,data_dir,loss_mode,desirable_class,log_loss,from_logits,smooth,ignore_index,eps):
    """
    Select and initialize a loss function based on the given parameters.

    Args:
        criterion_name (str): Name of the loss function to use ("DiceLoss" or "BCEWithLogitsLoss").
        data_dir (str): Base directory for the dataset, used to compute class weights if necessary.
        loss_mode (str): Mode for the Dice loss function (e.g., "binary" or "multiclass").
        desirable_class (int): Number of classes in the dataset.
        log_loss (bool): Whether to compute the logarithm of the Dice loss.
        from_logits (bool): Whether the input to BCEWithLogitsLoss is raw logits.
        smooth (float): Smoothing factor for the Dice loss to avoid division by zero.
        ignore_index (int): Label index to ignore in the Dice loss.
        eps (float): Small constant to avoid division by zero in the Dice loss.

    Returns:
        nn.Module: Initialized loss function object.
    """
    
    if criterion_name == "DiceLoss":
        criterion = smp.losses.DiceLoss(loss_mode,list(range(desirable_class)),log_loss,from_logits,smooth,ignore_index,eps)            
    
    if criterion_name ==  "BCEWithLogitsLoss":
        image_train_dir, mask_train_dir, image_val_dir, mask_val_dir=get_directories(data_dir)
        pos_weight = weight_tensor(mask_train_dir,desirable_class)
        criterion = nn.BCEWithLogitsLoss(pos_weight)
    
    if criterion_name == "JaccardLoss":
        criterion = smp.losses.JaccardLoss(loss_mode,list(range(desirable_class)),log_loss,from_logits,smooth,eps)

    if criterion_name == "FocalLoss":
        criterion = smp.losses.FocalLoss(loss_mode,ignore_index)
    
    return criterion