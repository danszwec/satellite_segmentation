import torch
import numpy as np
import matplotlib
import segmentation_models_pytorch as smp
import torch.nn as nn   
# matplotlib.use('TkAgg')
import random
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
#handle config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
with open('config.yaml', 'rt') as f:
        cfg = yaml.safe_load(f.read())

data_name = cfg['data']['name']



colormap = {0: [255, 0, 0],    # Red
    1: [0, 255, 0],    # Green
    2: [0, 0, 255],    # Blue
    3: [128, 0, 128],  # Purple
    4: [255, 0, 255],  # Magenta
    5: [0, 255, 255],  # Yellow
    6: [128, 128, 128], # Gray 
    7: [255, 165, 0],  # Orange
    8: [255, 255, 0],  # Cyan
    9: [0, 128, 0],     # Dark Green
    10: [0, 0, 128]    # Another Color 
}

def crop_image(image):
    """
    Crops an image to a 256x256 pixel area, centered on the original image.

    Args:
        image (PIL.Image.Image): The input image to be cropped. It must be a valid image object loaded using the Pillow library.

    Returns:
        PIL.Image.Image: The cropped image with a size of 256x256 pixels, centered around the middle of the original image.
    """
    width, height = image.size
    left = (width - 256) // 2
    top = (height - 256) // 2
    right = (width + 256) // 2
    bottom = (height + 256) // 2
    image = image.crop((left, top, right, bottom))
    return image 



def coloring_masks(tensor):
    """
    Converts a segmentation mask tensor into a color-encoded mask using a predefined colormap.

    Args:
        tensor (torch.Tensor or np.ndarray): A 2D or 3D tensor containing class labels for each pixel in the mask.
                                             If a 3D tensor is provided, the first dimension should have a size of 1 (i.e., [1, height, width]).
                                             The function can also accept a NumPy array, which will be converted to a torch tensor.

    Returns:
        torch.Tensor: A 3D tensor representing the color-encoded mask, with dimensions [3, height, width], where each pixel's color corresponds to its class.
                      The 3 channels correspond to the RGB values.

    Raises:
        ValueError: If the input tensor has more than 3 dimensions.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if type(tensor) == torch.Tensor:
        if tensor.dim()==3:
            tensor= tensor.squeeze(dim=0)
        if tensor.dim()>3:
            print("insert not the right dim")
    color_matrix = torch.zeros((3,tensor.shape[0],tensor.shape[0]))
    unique_classes = torch.unique(tensor)
    for clas in unique_classes:
        coordinates = (tensor == clas).nonzero() #return all the locations of the class
        for coord in coordinates:
            color_matrix[:,int(coord[0]),int(coord[1])] = torch.tensor(colormap[int(clas)])
    return color_matrix

def add_mask(original_image,mask):
    """
    Overlays a color-encoded mask on an original image by blending the two, with padding if necessary.

    Args:
        original_image (np.ndarray or PIL.Image.Image): The original image to which the mask will be applied.
                                                        It should be either a NumPy array or an image loaded using Pillow.
        mask (torch.Tensor): A 3D tensor representing the color-encoded mask with dimensions [3, height, width].
                             The mask will be blended with the original image.

    Returns:
        np.ndarray: The blended image, where the mask has been applied on top of the original image.
                    The image is returned as a NumPy array with values in the range [0, 255].

    Raises:
        ValueError: If the original image and the mask have incompatible dimensions, padding will be applied to match sizes.
    """

    # Read the original image and the color mask
    # original_image = cv2.imread(origin_image)
    original_matrix = np.array(original_image)
    mask = (mask.permute(1, 2, 0)).cpu().numpy()
    if original_matrix.shape != mask.shape:
        pad = (mask.shape[0]-original_matrix.shape[0])//2
        original_matrix = np.pad(original_matrix, ((pad,pad), (pad, pad), (0, 0)) , mode='constant', constant_values=0)

    #color_mask = mask
    # Define the alpha value for blending (adjust as needed)
    alpha = 0.15

    # Blend the original image and the color mask
    blended_image = (1 - alpha) * original_matrix + alpha *mask
    # Ensure the blended image has valid pixel values in the range [0, 255]
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image


def class_reduction(mask,new_class):
    """
    Reduces the number of classes in a segmentation mask by combining several classes into fewer classes.
    warning! the combined depended on data set. for custom datac set insert which classes you want to merge
    Args:
        mask (np.ndarray): A 2D array representing the segmentation mask with class labels.
        new_class (int): The number of desired classes after reduction.
                         Determines how classes are combined:
                         - 4: Combine classes based on specific rules (for "landcover" dataset).
                         - 3: Combine classes into 3 classes, with special handling for class 5.
                         - 2: Combine all classes into 2 classes, with all classes except class 1 being set to 0.
        data_name (str): The name of the dataset, used to apply specific class reduction rules if `new_class` is 4.

    Returns:
        np.ndarray: The updated mask with reduced classes.

   
    """
    mask = np.array(mask)
    if new_class == 4:
        if data_name == "landcover":
             mask[mask == 2] = 0 #agriculture to unknown
             mask[mask == 5] = 0  #water to unknown
             mask[mask == 6] = 3 #rangeland to barren_land
             mask[mask == 4] = 2 #forest is class number 2


        else:     
            for i in range(2,new_class+1):
                mask[mask == i] = 1  
    if new_class == 3:
            for i in range(2,7):
                if i != 5:
                    mask[mask==i] = 0
            mask[mask==5] = 2
            
    if new_class == 2:
        for j in range(7):
            if j!=1:
                mask[mask==j] = 0
    else:
        pass
    return mask


def channel_class(mask,new_class):
    """
    Converts a class label mask into a one-hot encoded tensor.

    Args:
        mask (np.ndarray): A 2D array or tensor representing the class labels. The shape should be [height, width] or 
                           [batch_size, height, width]. Each pixel value corresponds to a class label.
        new_class (int): The total number of classes in the mask. Determines the number of channels in the one-hot encoded tensor.

    Returns:
        torch.Tensor: A 3D tensor with shape [new_class, height, width], where each channel represents one class. 
                      The tensor contains one-hot encoded values for each class label.

    Notes:
        - The function assumes that the input mask contains class labels from 0 to `new_class - 1`.
        - The resulting one-hot tensor has `new_class` channels, each of size [height, width].
    """
    # Replace with the actual number of classes
    mask = np.array(mask)
    # Get the dimensions of the mask
    height,width = mask.shape[-2],mask.shape[-1]

    # Initialize a tensor for one-hot encoding
    one_hot = torch.zeros((new_class , height, width), dtype=torch.float32)

    # Fill the one-hot tensor
    for c in range(new_class):
        one_hot[c] = torch.tensor(mask == c, dtype=torch.float32)
    return one_hot

def transform_image(image,lst):
    """
    Applies a series of random image transformations to an input image, with normalization and optional random rotation.

    Args:
        image (PIL.Image.Image or torch.Tensor): The input image to be transformed. It can be a PIL image or a PyTorch tensor.
        lst (list of torchvision.transforms): A list of torchvision transform objects. Each transform is randomly applied with a 50% chance.
                                              If a `torchvision.transforms.RandomRotation` is in the list, its angle is chosen randomly between 0 and 360 degrees.

    Returns:
        torch.Tensor: The transformed image as a PyTorch tensor, normalized and with applied transformations.
        list: The list of transformations that were applied to the image, including the normalization and any random rotations.

    Notes:
        - The function applies normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
        - The transformations in `lst` are only applied with a 50% probability.
        - If a `RandomRotation` transform is included in `lst`, its angle is set to a random value between 0 and 360 degrees.
    """
    image_lst = [T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    for i in range(len(lst)):
        random_bol = random.randint(0, 1)
        if random_bol:
            if isinstance(lst[i] , T.RandomRotation):
                p = random.randint(0, 360)
                lst[i] = T.RandomRotation((p,p))
            image_lst.append(lst[i])
    curr_transform = T.Compose(image_lst)
    image = curr_transform(image)
    return image , image_lst   

def transform_mask(mask,transforms_list):
    """
    Applies a series of transformations to a segmentation mask, excluding transformations that are not suitable for masks.

    Args:
        mask (np.ndarray or PIL.Image.Image): The input mask to be transformed. It can be either a NumPy array or a PIL image.
        transforms_list (list of torchvision.transforms): A list of torchvision transform objects. Transformations that are not suitable for masks
                                                          (e.g., `ColorJitter`, `Normalize`, `ToTensor`) are filtered out.

    Returns:
        PIL.Image.Image or torch.Tensor: The transformed mask. If the input mask was a NumPy array, the output will be a PyTorch tensor.
                                         Otherwise, it will be a PIL image.

    Notes:
        - Transformations that are typically applied to color images but not to masks (like normalization or color jitter) are removed from the list.
        - The resulting mask is transformed using only the remaining transformations.
    """
    types_to_remove = (T.ColorJitter,T.Normalize,T.ToTensor)
    filtered_transforms = [item for item in transforms_list if not isinstance(item, types_to_remove)]
    curr_transform = T.Compose(filtered_transforms)
    mask = curr_transform(mask)
    return mask




def image_vs_mask(images,masks,batch_size):
    """
    Visualizes a batch of images alongside their corresponding masks.

    Args:
        images (torch.Tensor): A 4D tensor containing a batch of images with shape `[batch_size, channels, height, width]`.
        masks (torch.Tensor): A 4D tensor containing the corresponding masks with shape `[batch_size, num_classes, height, width]`.
        batch_size (int): The number of images and masks to display from the batch.

    Returns:
        None

    Notes:
        - Assumes that the images are normalized with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
        - The function denormalizes the images before plotting.
        - Uses the `coloring_masks` function to convert mask tensors into color-encoded images.
        - Displays the images and masks in a side-by-side layout using matplotlib.
    """
    for i in range(batch_size):
        image = images[i,:,:,:]
        mask = (masks[i,:,:,:]).argmax(dim=0)

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        mean = mean.to(image.device)
        std = std.to(image.device)
        
        # Denormalize
        image = image * std[:, None, None] + mean[:, None, None]


        image = image.permute(1, 2, 0).cpu().numpy()
        mask = coloring_masks(mask)
        mask = mask.permute(1, 2, 0).cpu().numpy()
        # Plot the images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image)
        axes[0].set_title('Image 1')
        axes[0].axis('off')

        axes[1].imshow(mask)
        axes[1].set_title('Image 2')
        axes[1].axis('off')

        plt.show()
        print("a")

def tensor_to_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized tensor back to a PIL Image with pixel values in [0, 255].

    Parameters:
        tensor (torch.Tensor): The normalized tensor to convert.
        mean (list or tuple): Mean values used for normalization.
        std (list or tuple): Standard deviation values used for normalization.

    Returns:
        PIL.Image: The denormalized and converted PIL Image.
    """
    # Ensure the tensor has the same number of channels as the mean and std
    if tensor.size(0) != len(mean):
        raise ValueError("Tensor channels do not match the length of mean and std lists")

    # Reverse normalization
    mean = torch.tensor(mean).reshape(-1, 1, 1)
    std = torch.tensor(std).reshape(-1, 1, 1)
    denormalized_tensor = tensor * std + mean
    
    # Clip tensor values to the [0, 1] range and scale to [0, 255]
    denormalized_tensor = denormalized_tensor.clip(0, 1)  # Ensure values are within [0, 1]
    denormalized_tensor = (denormalized_tensor * 255).byte()  # Scale to [0, 255] and convert to byte

    # Convert tensor to PIL image
    return denormalized_tensor
