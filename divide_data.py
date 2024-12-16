import os
import random

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
    