import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import utils.data_utils
import yaml

#handle config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
with open('config.yaml', 'rt') as f:
        cfg = yaml.safe_load(f.read())
data_name = cfg['data']['name']

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





def one_hot(masks,desirable_class):
    """
    Converts a segmentation mask with class labels into a one-hot encoded tensor with a specified number of classes.

    Args:
        mask (torch.Tensor): A 3D tensor representing the segmentation mask with class labels.
        new_class (int): The number of classes for the one-hot encoding.

    Returns:
        torch.Tensor: A 4D tensor representing the one-hot encoded mask, with dimensions [batch_size, num_classes, height, width].
    """

    # Ensure the mask is of type Long for one-hot encoding
    masks = masks.long()  # [batch_size, 1, H, W]

    masks = masks.squeeze(1)  # [batch_size, H, W]

    # Apply one-hot encoding
    one_hot = torch.nn.functional.one_hot(masks, num_classes=desirable_class)  # [batch_size, H, W, num_classes]

    # Convert to float and rearrange dimensions to [batch_size, num_classes, H, W]
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    return one_hot

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


def grey_to_rgb_mask(model, input, path = None):

    # Get the model output
    output = utils.train_utlis.predict(model, input)
    output = output.cpu().numpy()

    # Load the color dictionary
    if path:
        color_dict = utils.data_utils.parse_color_file(path)
    else:
        color_dict = utils.data_utils.parse_color_file()

    # Create an empty RGB mask
    rgb_mask = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)

    # Convert the grayscale mask to an RGB mask
    for index,color in enumerate(color_dict.values()):
        rgb_mask[output == index] = color

    return rgb_mask

def visualize_comparison(pred, target,description = None):
    # Convert the input and target to numpy arrays
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    #plot the input and target side by side
    plt.figure(figsize=(10, 5))

    #first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(pred)
    plt.title('prediction')
    plt.axis('off')

    #second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(target)
    plt.title('ground truth')
    plt.axis('off')

    #show the plot
    plt.tight_layout()
    plt.show()
    
    #save the plot
    plt.savefig(f'comparison_{description}.png')

    return 


##### cvat convertor #####



def rgb_to_grey(mask,label_dict):
    # Create an empty grayscale mask
    grey_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    
    # Convert the RGB mask to a grayscale mask
    for index, color in enumerate(label_dict.values()):
        grey_mask[np.all(mask == color)] = index
        
    return grey_mask