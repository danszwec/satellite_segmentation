import torch
import torch.nn.functional as F

def predict(model, input):
    output = model(input)
    output = F.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    return output
