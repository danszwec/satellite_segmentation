import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_confusion_matrix_with_metrics(resutls, model_path):
    """
    Function to plot the confusion matrix and display evaluation metrics in a fancy style above it.
    
    Parameters:
    - model_confusion_matrix: The confusion matrix (2D NumPy array) resutls[0]
    - pixel_accuracy: Pixel accuracy value (float) resutls[1]
    - iou_micro: IoU (micro) value (float) resutls[2]
    - iou_weighted: IoU (weighted) value (float)    resutls[3]
    - lowest_iou: The lowest IoU per class (float) resutls[4]
    - recall: Recall values per class (NumPy array)     resutls[5]
    - precision: Precision values per class (NumPy array) resutls[6]
    - f1_score: F1 score values per class (NumPy array) resutls[7]
    - model_path: Path to save the evaluation metrics plot image (string)
    """
    # Set the style of the plot
    sns.set(style="whitegrid")

    # Create the figure for the confusion matrix
    plt.figure(figsize=(12, 10))

    # Plot the confusion matrix
    sns.heatmap(resutls[0]  , annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Number of Pixels'}, annot_kws={"size": 14, "weight": "bold"})

    # Add title and labels
    plt.title('Confusion Matrix', fontsize=18, weight='bold')
    plt.xlabel('Predicted', fontsize=14, weight='bold')
    plt.ylabel('True', fontsize=14, weight='bold')

    # Prepare the metrics text in a clean format
    metrics_text = f"""
    Pixel Accuracy:        {resutls[1]:.4f}
    IoU (Micro):           {resutls[2]:.4f}
    IoU (Weighted):        {resutls[3]:.4f}
    Lowest IoU per Class:  {resutls[4]:.4f}
    Recall:                {resutls[5]:.4f}
    Precision:             {resutls[6]:.4f}
    F1 Score:              {resutls[7]:.4f}
    """

    # Add the metrics text above the confusion matrix
    plt.gcf().text(0.1, 0.85, metrics_text, fontsize=14, weight='bold', color='darkblue', ha='left')

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plot_path = model_path.replace('.pth', 'evaluation_metrics.png')
    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")
    return  



def compare_models_performers(models_dict, save_dir):
    """
    Compare items in the lists of each key in the dictionary and plot all comparisons 
    in the same figure using subplots. Save the plot in the specified directory.

    Parameters:
    - models_dict: Dictionary where keys are model names/identifiers, and values are lists of items to compare.
    - save_dir: Directory where the plot will be saved.
    """
    # Ensure the dictionary is not empty
    if not models_dict:
        print("The dictionary is empty!")
        return
    
    # Get the number of items in each list (assuming all lists are the same length)
    list_length = len(next(iter(models_dict.values())))
    
    # Get the keys (model names)
    keys = list(models_dict.keys())
    
    # Create a figure with subplots
    fig, axes = plt.subplots(list_length, 1, figsize=(8, 6 * list_length))
    if list_length == 1:  # If there is only one item to compare, axes will not be an array
        axes = [axes]
    
    # Iterate over each index (item position) in the lists
    for i in range(list_length):
        # Collect values from each key's list at the current index
        values = [models_dict[key][i] for key in keys]
        
        # Plot the comparison as a bar graph in the corresponding subplot
        axes[i].bar(keys, values, color='skyblue')
        
        # Add title and labels to each subplot
        axes[i].set_title(f'Comparison of Item {i+1} Across Models', fontsize=14)
        axes[i].set_xlabel('Models', fontsize=12)
        axes[i].set_ylabel(f'Value of Item {i+1}', fontsize=12)

    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'comparison_plot.png')
    plt.savefig(plot_path)
    
    # Optionally, print the save path for reference
    print(f"Plot saved at: {plot_path}")
    return

