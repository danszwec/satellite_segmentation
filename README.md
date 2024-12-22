# <div align="center">Satellite Image Semantic Segmentation</div>

## Project Overview
This project focuses on semantic segmentation of aerial/satellite imagery for land cover classification. The model processes RGB satellite images (256x256x3) in PNG format to identify and segment different land cover features. The project evaluates various deep learning architectures (such as **DeepLabV3+**, **UNet**, etc.) to determine the most effective approach for high-resolution satellite image segmentation, with applications in urban, forest, water and other terrain features detection.

<details open>
  <summary><strong>Pre-trained project details:</strong></summary>
<br>
While the project structure is modular and can accommodate different architectures and datasets, this specific implementation includes:
  
* A trained model using the **DeepLabV3+** architecture
* Training performed on a land cover dataset (included in the repository)
* Optimization using **AdamW** optimizer
* Training with **Dice loss** function for semantic segmentation
</details>

## Table of Contents
- [Getting Started: Important Requirements and Labeling, Project Directory Structure](#getting-started-important-requirements-and-labeling)
- [Data Organization and Optional Dataset](#data-organization-and-optional-dataset)
- [Installations](#installations)
- [Model Selection, Loss Functions, and Optimizers](#Model-selection-loss-functions-and-optimizers)
- [Usage](#usage)
  - [Config file](#config-file)
  - [How to use main.py](#how-to-use-main)
- [Pre-trained Model Results](#pre-trained-model-results)

## Getting Started: Important Requirements and Labeling
- Ensure that the images and masks is organized and as written in "Data organinzation and Optional Dataset".
- Ensure that eace image has a corresponding mask
- All images and masks should be of the same resolution (e.g., **256x256**).
- Images should be in formats such as **PNG** or **JPEG**, and masks should be **grayscale** images where each pixel corresponds to a class label (e.g., 0 for background, 1 for urban, 2 for forest, etc.).
- **If you want to tag images for this semantic segmentation project, please visit [The CVAT Labeling Infrastructure Guide](https://github.com/danszwec/segmentation-labeling-Toolkit), where you'll find instructions on how to tag and save the images.**

### Project Directory Structure:
To set up the project environment:
1. Clone the repository to your desired location:
```bash
git clone https://github.com/danszwec/satellite_segmentation
cd satellite_segmentation
```
2. Navigate to your project directory and ensure it has the following structure:
```
your_model_path/
├── Data/
│   └── Dataset_dir/
├── result/          # Generated automatically
├── utils/
├── main.py
├── testEval.py
├── train.py
├── inference.py
└── config.yaml
```

## Data Organization and Optional Dataset
### Data Organization
The dataset should be organized in the following structure:

1. **`Dataset_dir/train/images/`**: Place your training images here.
2. **`Dataset_dir/train/masks/`**: Place the corresponding ground truth masks for training images here.
3. **`Dataset_dir/val/images/`**: Place your validation images here.
4. **`Dataset_dir/val/masks/`**: Place the corresponding ground truth masks for validation images here.
5. **`Dataset_dir/test/images/`**: Place your test images here.
6. **`Dataset_dir/test/masks/`**: Place the corresponding ground truth masks for test images here.

### Optional Dataset
place here the optional dataset that we work with.
* להעלות לפה את הקישור של הדאטה סט הזה, ולכתוב שזה צריך לעבור שינוי ל-256 קרופים.

## Installations:
**still need to update!!!!! download a docker**
* אי אפשר להעלות לגיט קובץ ענק של 6 ג'יגה אז אי אפשר להעלות לפה דוקר שלם (אפשר לעשות את זה בשביל המחשב שלנו)
*  לשים פה קישור לדוקר פייל
* להכין קובץ ריקוויירמנטס בשביל כל הפרוייקט

## Model Selection, Loss Functions, and Optimizers
### Model Architecture
This project offers several segmentation models:
* DeepLabV3+: Uses atrous convolution and a decoder for precise boundary segmentation.
* UNet: Classic encoder-decoder model with skip connections, effective for medical image tasks.
* PSPNet: Uses pyramid pooling for multi-scale context and improved scene understanding.
* UNet++: Extends UNet with nested skip pathways for better feature fusion and accuracy.

### Loss Functions
Available loss functions include:
* Dice Loss: Optimizes for overlap between predicted and ground truth.
* BCEWithLogitsLoss: Combines binary cross-entropy with a sigmoid layer.
* Jaccard Loss: Focuses on maximizing intersection-over-union (IoU).
* Focal Loss: Down-weights well-classified examples to focus on hard ones.

### Optimizers
Choose from the following optimizers:
* AdamW: Optimizer with weight decay, ideal for most tasks.
* Adam: Adaptive optimization combining RMSProp and momentum.
* SGD: Classic optimizer, effective for fine-tuning.

### Default Configuration
The default setup uses **DeepLabV3+** with **BCEWithLogitsLoss** and the **AdamW** optimizer, offering a good balance for general segmentation tasks.


## Usage

### config File
The config.yaml file contains all the essential settings for configuring your training process. By modifying these settings, you can fine-tune the training process to suit your dataset and desired outcomes.

**Key Settings to Modify for Effective Training**
Before running the test, you can adjust the following parameters in the config.yaml to optimize your training:
- Model: Choose the architecture (e.g., DeepLabV3+, UNet, etc.).
- Loss Function: Select a suitable loss function (e.g., BCEWithLogitsLoss, Dice Loss).
- Optimizer: Choose between AdamW, Adam, or SGD.
- Learning Rate: Adjust the learning rate for optimal training speed and stability.
- Batch Size: Modify the batch size to suit your GPU memory capacity.
- Epochs: Set the number of epochs based on the size of your dataset and training goals.

**Make sure the settings align with your hardware capabilities and dataset size for efficient training.**

There are two options for modifying the config.yaml file:
1. Go to the config.yaml file, change the settings, and save it (this will become your default configuration).
2. Use the menu and choose not to use the default configuration (this will be a temporary configuration). If you like the outcome of the model after the run, you can set it as the default at the end.

### How to use main
**In main.py - Follow the prompts on-screen to ensure a smooth workflow.**

1. **Launch the Script:** Run the script from your terminal or command line interface.
2. **Welcome Screen:**
   - You will be presented with a welcome screen.
   - Select 1 to initiate the model training process.
3. **Training Workflow Environment:**
   - You will enter the training workflow environment.
   - Choose whether to use the default pre-trained configuration:
     - Y: Load the default configuration.
     - n: Enter the configuration update screen.

**Configuration Options:** In the configuration update screen, customize the following:
- Architecture: Select from the following options:
  * DeepLabV3+
  * Unet
  * PSPNet
  * Unet++
- Loss Function: Choose one of the following:
  * Dice Loss
  * BCE With Logits Loss
  * Jaccard Loss
  * Focal Loss
- Optimizer: Select an optimizer:
  * AdamW
  * Adam
  * SGD
- Weight Decay: Choose a weight decay value:
  * 0.001
  * 0.0001
  * 0.00001
4. **Dataset Configuration:**
   - You will be asked if you want to change the dataset:
     - Y: Specify the path to your dataset and provide a name.
     - n: Continue using the current dataset.
5. **Additional Settings:**
   - Choose the batch size for training.
   - Specify the number of epochs.

After finalizing all configurations, you will see a summary of the selected settings, and the training loop will begin automatically.

## Pre-trained Model Results


























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
