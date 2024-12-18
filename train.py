# Define the Training Loop
import os
from colorama import Fore, Style, init
from tqdm.auto import tqdm
from tqdm import tqdm
import sys
from data_utilz import * 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from define_datasetclass import SegmentationDataset 
import matplotlib.pyplot as plt
from utilz import *
from utils.loss_utils import *
from image_utils import image_vs_mask
torch.backends.cudnn.enabled = False
from datetime import datetime
import yaml
from utils.cfg_utils import *

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"  
date = str(datetime.now().strftime("%Y-%m-%d"))


def train(cfg): #Pull all the vars from the config file
    #cfg
    data_dir = cfg['data']['dir']

    #train
    criterion_name = cfg['loss']['name']
    desirable_class = cfg['train']['desirable_class']
    batch_size = cfg['train']['batch_size']
    optimizer_name = cfg['train']['optimizer_name']
    lr = cfg['train']['lr']
    weight_decay = cfg['train']['weight_decay']
    epslion = cfg['train']['check_convergence_epslion']
    num_epochs = cfg['train']['num_epochs']
    back_epochs = cfg['train']['back_epochs']

    #loss
    loss_mode = cfg['loss']['mode']
    log_loss = cfg['loss']['log_loss']
    from_logits = cfg['loss']['from_logits']
    smooth = cfg['loss']['smooth']
    ignore_index = cfg['loss']['ignore_index']
    eps = cfg['loss']['eps']

    #transform
    transform_dict = cfg['transformes']['types']
    p = cfg['transformes']['possibility']
    degrees = cfg['transformes']['degrees']

    #model
    model_name = cfg['model']['model_name']
    encoder_weights = cfg['model']['encoder_weights']
    encoder_name = cfg['model']['encoder_name']
    activation = cfg['model']['activation']
    pooling = cfg['model']['pooling']
    dropout = cfg['model']['dropout']

####################################################################################

    # Create a directory for the train
    save_dir = train_dir(model_name)
    checkpoints_dir = os.path.join(save_dir,'checkpoint')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # load the data
    train_loader, val_loader= load_data(cfg,desirable_class,batch_size,data_dir)

    # Initialize the model, loss function, and optimizer
    model = load_model(cfg)   
    optimizer = select_optimizer(model,optimizer_name,lr,weight_decay)
    criterion = select_loss(criterion_name,data_dir,loss_mode,desirable_class,log_loss,from_logits,smooth,ignore_index,eps)
    
    # Training loop
    num_iter = len(train_loader)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    model.to(device)
    bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)
    bar_format1 = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)
    with tqdm(total=num_epochs, desc="Training Progress",ncols=150, unit='epoch',bar_format=bar_format) as epoch_bar:
        for epoch in range(num_epochs):
            loss_val = 0
            acc_val = 0    
            with tqdm(total=num_iter, desc="batch Progress",ncols=100  , unit='iter',bar_format=bar_format1) as iter_bar:
                # Training step
                model.train()
                acc_train= 0 
                batch_loss = 0.0
                for images, masks in train_loader:
                    torch.cuda.empty_cache()    
                    images, masks = images.to(device), masks.to(device)                    
                    
                    # Forward pass1
                    outputs = model(images)[0]
                    one_hot_mask = one_hot(masks,desirable_class)
                    loss = criterion(outputs, one_hot_mask)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Calculate accuracy and loss of iter
                    item_accuracy = get_accuracy(outputs,masks,desirable_class)
                    
                    # Calculate accuracy and lose of epoch
                    batch_loss += loss.item() * images.size(0) # multiply batch loss by the size batch
                    acc_train += item_accuracy*images.size(0) # same as the loss
                    iter_bar.update(1) #update

            # Calculate the loss and accuracy of the epoch
            epoch_loss = batch_loss / len(train_loader.dataset) # divide the loss by all the data set
            epoch_acc = acc_train/len(train_loader.dataset)       # same as loss
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
        
            # Validation step
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)[0]
                    masks = one_hot(masks,desirable_class)
                    
                    # Calculate accuracy and loss of the validation
                    loss = criterion(outputs, masks)
                    loss_val += loss.item() * images.size(0)
                    acc_val += get_accuracy(outputs,masks,desirable_class)* images.size(0)
                    val_loss = loss_val / len(val_loader.dataset)
                    val_acc = acc_val/len(val_loader.dataset)

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # print the processes
            sys.stdout.flush()
            print("\naccuracy train:" ,epoch_acc , "loss train:" ,epoch_loss , "\n" "accuracy val:" , val_acc," loss val:",val_loss)
            epoch_bar.update()
            
            #plot curves
            update_learning_curves(train_accuracies, val_accuracies, train_losses,val_losses ,num_epochs,epoch ,model_name,save_dir)            
            
            # Save the model
            if epoch % 10 == 0:
                # Save the model
                torch.save(model.state_dict(),os.path.join(checkpoints_dir,f'{model_name}_epoch_{epoch}.pth'))

                # if the training is converged , stop the training
                if check_convergence(train_losses,val_losses,back_epochs,epslion):
                    break
    return checkpoints_dir

if __name__ == "__main__":
    yaml_file = '/workspace/config.yaml'
    cfg = load_yaml(yaml_file)
    train(cfg)