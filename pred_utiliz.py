import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from cvat_convertor import parse_color_file

def predict(model, input):
    output = model(input)
    output = F.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    return output

def grey_to_rgb_mask(model, input, path = None):

    # Get the model output
    output = predict(model, input)
    output = output.cpu().numpy()

    # Load the color dictionary
    if path:
        color_dict = parse_color_file(path)
    else:
        color_dict = parse_color_file()

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