import os
import numpy as np
from PIL import Image
from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')

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

def rgb_to_grey(mask,label_dict):
    # Create an empty grayscale mask
    grey_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    
    # Convert the RGB mask to a grayscale mask
    for index, color in enumerate(label_dict.values()):
        grey_mask[np.all(mask == color)] = index
        
    return grey_mask

def fetch_masks(base_dir):
    # where the masks are located
    masks_path = os.path.join(base_dir, 'SegmentationClass')

    # create a directory to save the annotations
    annotations_dir = os.path.join(today ,'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    # Get the list of files
    masks_files = os.listdir(masks_path)

    # Get the color dictionary
    label_dict =parse_color_file(base_dir)

    # Convert the masks to grayscale and save them
    for file in masks_files:
        mask = np.array(Image.open(os.path.join(masks_path, file)))
        grey_mask = Image.fromarray(rgb_to_grey(mask, label_dict))
        grey_mask.save(os.path.join(annotations_dir, file))
    return masks_files

def fetch_images(base_dir,masks_files):
    # where the images are located
    images_path = os.path.join(base_dir, 'JPEGImages')

    # create a directory to save the images
    images_dir = os.path.join(today ,'images')
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

if __name__ == '__main__':
    pull_data('data')
    
        

