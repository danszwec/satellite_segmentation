# Satellite Image Semantic Segmentation

## Project Overview
This project focuses on semantic segmentation of aerial/satellite imagery for land cover classification. The model processes RGB satellite images (512x512x3) in PNG format to identify and segment different land cover features. The project evaluates various deep learning architectures (such as **DeepLabV3+**, **UNet**, etc.) to determine the most effective approach for high-resolution satellite image segmentation, with applications in urban, forest, water and other terrain features detection.

While the project structure is modular and can accommodate different architectures and datasets, this specific implementation includes:
- A trained model using the **DeepLabV3+** architecture
- Training performed on a land cover dataset (included in the repository)
- Optimization using **AdamW** optimizer
- Training with **Dice loss** function for semantic segmentation

## Table of Contents
- Getting Started: Important Requirements and Labeling, Peoject Directory Structure.
- Data Organization and Optional Dataset
- Installation
- Model Architecture
- Usage
- Pre-trained Model Results
- Acknowledgments

## Important Requirements and Labeling
- Ensure that each image in the `images` folder has a corresponding mask in the `masks` folder.                 (- Ensure that the images is organized as written in "Data organinzation and Optional Dataset" to make your run easier)
- All images and masks should be of the same resolution (e.g., **256x256** or **512x512**).
- Images should be in formats such as **PNG** or **JPEG**, and masks should be grayscale images where each pixel corresponds to a class label (e.g., 0 for background, 1 for urban, 2 for forest, etc.).

### Project Directory Structure:
To set up the project environment:
1. Clone the repository to your desired location:
   git clone <repository-url>
2. Navigate to your project directory and ensure it has the following structure:
   your_model_path/
  ├── Data/
  │   └── Dataset_dir/
  ├── result/          # Generated automatically
  ├── utils/
  ├── main.py
  ├── testEvaluation.py
  ├── train.py
  ├── inference.py
  └── config.yaml

## Data Organization and Optional Dataset
### Data Organization
The dataset should be organized in the following structure:

1. **`Dataset_dir/train/images/`**: Place your training images here.
2. **`Dataset_dir/train/masks/`**: Place the corresponding ground truth masks for training images here.
3. **`Dataset_dir/val/images/`**: Place your validation images here.
4. **`Dataset_dir/val/masks/`**: Place the corresponding ground truth masks for validation images here.
5. **`Dataset_dir/test/images/`**: Place your test images here.
6. **`Dataset_dir/test/masks/`**: Place the corresponding ground truth masks for test images here.

**Requirements:**
- All images must be in 512x512 resolution, RGB format, and saved as PNG/JPEG files
- Each image MUST have a corresponding mask file. All masks must match the image resolution and be single-channel
- Each pixel in the mask represents an integer value for a specific class label (e.g., 0 for background, 1 for urban, etc.)

### Optional Dataset
place here the optional dataset that we work with.





Installation
bashCopy# Clone the repository
git clone [YOUR_REPOSITORY_URL]

# Install dependencies
pip install -r requirements.txt
Dataset
[Please provide information about:]

Dataset source
Number of images
Classes/features being segmented
Data split (train/validation/test)
Image specifications:

Resolution: 512x512
Channels: 3 (RGB)
Format: PNG



Model Architecture
[Please specify:]

Base architecture: DeepLabV3+
Optimizer: AdamW
Loss function: Dice Loss
Any modifications made
Number of layers
Training hyperparameters

Usage
Training
pythonCopy# Example training command
python train.py --data_path /path/to/dataset --epochs 100
Inference
pythonCopy# Example inference command
python predict.py --image_path /path/to/image.png --model_path /path/to/saved_model
Pre-trained Model Results
[Please provide:]

Performance metrics of the pre-trained model
Validation/test set results
Sample segmentation outputs
Training convergence graphs
Comparison with baseline models (if applicable)

Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
[Please specify the license type]
Contact
[Your contact information or preferred method of contact]
Acknowledgments

[List any acknowledgments, papers referenced, or tools used]








הפעלה דרך ה-מיין.
משם- שולח אותך לבחור איזה yaml אתה רוצה. את הרגיל שהוא זה שהיה עליו את כל המודל שלנו, או להגדיר את שלך.
אח"כ האם תרצה להריץ אימון או ישר לעשות אלווציה.
אם אתה מרוצה - האם לשמור את הקונפיגורציה?

צריך גם לכתוב האם צריך להכין תיקיות מסויימות (נגיד למודלים אם צריך לאמן או שזה מייצר לבד. האם צריך לשמור במקום מסויים את כל הקבצים או לסדר אותם איכשהו במחשב.


אח"כ- להמשיך את מה שכתוב כאן.

לגבי לולאת ה-train: לכתוב הסברים למה שרשום שם ומה צריך/ניתן לשנות. לא לפרט על כל שלב ולא על ה-utils שלו.

להסביר על תיקיית היוטילז ועל החלוקה של התיקייה בפנים (ל-מיין, לטריין, לדאטהסט פרפייר, לאוולואציה..)
אם יש עוד- לכתוב. (רק ראשי פרקים: התיקייה יוטילז מכילה תתי תיקיות של.. כל אחת מהן מכילה פונקציות של....)







































########################################################################


# Evaluating Segmentation Models for Satellite Data

## Description
This project aims to evaluate different segmentation models for land cover classification using satellite imagery. The focus is to determine the best-performing model for segmenting high-resolution satellite images based on various deep learning architectures. The project compares models such as **DeepLabV3+**, **U-Net**, and other segmentation models to assess their effectiveness in segmenting land cover types, including urban, forest, water, and other terrain features.

## Table of Contents
- [Description](#description)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Project Structure](#project-structure)
  - [Labeling Process for Custom Data](#labeling-process-for-custom-data)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)


## Project Structure

### Labeling Process for Custom Data
refer to the repository that explains how to label your own dataset

https://github.com/danszwec/segmentation-labeling-Toolkit.git

### model training

#### Data Organization

To use the training loop, please organize your dataset in the following structure:


##### Folder Breakdown:

1. **`train/images/`**: Place your training images here.
2. **`train/masks/`**: Place the corresponding ground truth masks for training images here.
3. **`val/images/`**: Place your validation images here.
4. **`val/masks/`**: Place the corresponding ground truth masks for validation images here.
5. **`test/images/`**: Place your test images here.
6. **`test/masks/`**: Place the corresponding ground truth masks for test images here.

##### Important Notes:
- Ensure that each image in the `images` folder has a corresponding mask in the `masks` folder.
- All images and masks should be of the same resolution (e.g., **256x256** or **512x512**).
- Images should be in formats such as **PNG** or **JPEG**, and masks should be grayscale images where each pixel corresponds to a class label (e.g., 0 for background, 1 for urban, 2 for forest, etc.).

## Installation

### 1. Clone the repository:
Is recommended set Up a Virtual Environment
```bash
git clone https://github.com/yourusername/satellite-segmentation.git
cd satellite-segmentation
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
## Usage

To start training the segmentation model, ensure that your data is organized as specified above and dependencies are installed.

#### Changing Hyperparameters

If you want to adjust the hyperparameters (e.g., learning rate, optimizer, batch size, or number of epochs) for model training, you will need to rewrite the `config.yaml` file in the cloned repository. 

After cloning the repository, follow these steps to update the configuration:

1. Open the `config.yaml` file in the root directory of the cloned repository.
2. Modify the parameters as needed and save.
#### Training the Model

Run the following command to begin training:

```bash
python train.py --model desire_model #Unet/Deeplabv3+
```
