import os
import shutil
from PIL import Image


def split_image(image_path, target_size=(512, 512)):
    """
    Splits an image into smaller patches of the specified size and saves them in a temporary directory.

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

    # Split the image into patches
    for y in range(0, height, target_size[1]):
        for x in range(0, width, target_size[0]):
            # Crop patch
            box = (x, y, min(x + target_size[0], width), min(y + target_size[1], height))
            patch = image.crop(box)

            # Save patch
            patch_filename = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{x}_{y}.png")
            patch.save(patch_filename)

    return temp_dir, (width, height)


def rebuild_image(temp_dir, original_size, image_path):
    """
    Rebuilds the original image from patches in a temporary directory and deletes the temporary directory.

    Args:
    - temp_dir (str): Path to the temporary directory containing the patches.
    - original_size (tuple): The original size of the image (width, height).
    - image_path (str): Path to the original image.

    Returns:
    - str: Path to the rebuilt image.
    """
    # Create a new blank image
    rebuilt_image = Image.new("RGB", original_size)

    # Read patches and paste them into the blank image
    for patch_name in sorted(os.listdir(temp_dir)):
        patch_path = os.path.join(temp_dir, patch_name)
        patch = Image.open(patch_path)

        # Extract coordinates from the filename
        _, x, y = os.path.splitext(patch_name)[0].rsplit("_", 2)
        x, y = int(x), int(y)
        rebuilt_image.paste(patch, (x, y))

    # Save the rebuilt image
    rebuilt_image_path = os.path.join(
        os.path.dirname(image_path), f"{os.path.splitext(os.path.basename(image_path))[0]}_rebuilt.png"
    )
    rebuilt_image.save(rebuilt_image_path, "PNG")

    # Cleanup: Delete the temporary directory and its contents
    shutil.rmtree(temp_dir)

    return rebuilt_image_path


# Example Usage
if __name__ == "__main__":
    # Input image path
    input_image = "/home/oury/Downloads/danhaefes/masks/default/14397_sat_forest_land_00000006.png"

    # Step 1: Split the image into patches
    temp_dir, original_size = split_image(input_image)

    # Step 2: Rebuild the original image from patches
    rebuilt_image_path = rebuild_image(temp_dir, original_size, input_image)

    print(f"Rebuilt image saved at: {rebuilt_image_path}")
