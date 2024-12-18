import os
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms as T

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
    filtered_transforms.append(T.PILToTensor())
    curr_transform = T.Compose(filtered_transforms)
    mask = curr_transform(mask)
    return mask



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

        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
    
        # Apply transformations
        image,transforms_list = transform_image(image,self.transform)
        mask = transform_mask(mask,transforms_list)
        return image, mask
    


