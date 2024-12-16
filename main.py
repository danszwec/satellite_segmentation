from colorama import Fore, init
init()
from train import train
from testEvaluation import test_evaluation
from utils import *
import yaml
from utils.cfg_utils import *
from pred_utiliz import *
from utilz import *
from image_handeling import *
from divide_data import *


if __name__ == "__main__":

    yaml_file = '/workspace/config.yaml'
    cfg = load_yaml(yaml_file)
    if cfg:
        print(f"{Fore.GREEN}Welcome to the segmentation model, \na modular model developed by the AI team in the Information Extraction department at ELTA System LTD.{Fore.RESET}")
        print(f"{Fore.GREEN}Model Interaction Menu:{Fore.RESET}")
        while True:    
            menu_ch = input(f"{Fore.CYAN}Please select your desired operation by entering the corresponding number.{Fore.RESET} \n1. Train Model: Initiate model training process \n2. Data Configuration: Define dataset parameters and preprocessing \n3. Pre-Trained Model Inference: Apply existing model for image segmentation \n4. exit \n")
            
            #Train Model Environment
            if menu_ch == "1":
                print(f"{Fore.GREEN}\nWelcome to our training workflow environment.{Fore.RESET}")
                default_cfg = input(f"{Fore.CYAN}Do you want to use the default pre-trained configuration?{Fore.RESET} \nY/n\n")
                if default_cfg == "Y":
                    print(f"{Fore.GREEN}Using default configuration..{Fore.RESET}")
                    data_and_trainloop_cfg(cfg)                                     # Update the dataset and the train loop
                    print_sum(cfg)
                if default_cfg == "n":
                    cfg = update_cfg(cfg)  
                train(cfg)                                                          # Start the train
                print(f"{Fore.GREEN}Training has been completed successfully.{Fore.RESET}\n{Fore.RED}Starting to evaluate the model..{Fore.RESET}")
                test_evaluation(train_dir,model,cfg)                               # evaluate the model weights by graphs parameters
                print(f"{Fore.GREEN}Evaluation has been completed successfully.{Fore.RESET}")
                save_yaml = input(f"{Fore.CYAN}Do you want to save the current configuration?{Fore.RESET} \nY/n\n")
                if save_yaml == "Y":
                    save_yaml(yaml_file,cfg)
                    print(f"{Fore.GREEN}The configuration has been saved successfully.{Fore.RESET}")
                    print(f"{Fore.GREEN}Finished the training process.{Fore.RESET}\n")
                if save_yaml == "n":
                    print(f"{Fore.GREEN}Finished the training process.{Fore.RESET}\n")
                continue
                
            #Data Configuration Environment
            if menu_ch == "2":         
                print(f"{Fore.GREEN}Welcome to the data configuration environment. \nThis Environment takes the data and organize it that it will fit to the model{Fore.RESET}")
                print(f"{Fore.RED}Important! The images and masks must be in 2 separate folders (images, masks) in the same directory!{Fore.RESET}\n")
                data_ch = input(f"{Fore.CYAN}Whould you like to:{Fore.RESET} \n1. Create a new dataset directory \n2. Add data to an existing dataset directory \n")
                # divide the data and save it in the new directory
                if data_ch == "1":
                    data_path = input(f"{Fore.CYAN}Enter the path of the data you want to organize:{Fore.RESET} \n")
                    new_dir_path = input(f"{Fore.CYAN}Enter the path for the new directory:{Fore.RESET} \n")
                    divide_data(data_path, new_dir_path)                                 # Divide the data to train, val and test
                    print(f"{Fore.GREEN}The data has been successfully divided and saved in {new_dir_path}.{Fore.RESET}\n")
                    print(f"{Fore.GREEN}Finished the data configuration process.{Fore.RESET}\n")
                # divide the data and save it in the existing directory
                if data_ch == "2":
                    data_path = input(f"{Fore.CYAN}Enter the path of the data you want to organize:{Fore.RESET} \n")
                    exist_dir = input(f"{Fore.CYAN}Enter the path for the existing directory:{Fore.RESET} \n")
                    divide_data(data_path, exist_dir)                                    # Divide the data to train, val and test
                    print(f"{Fore.GREEN}The data has been successfully divided and saved in {exist_dir}.{Fore.RESET}\n")
                    print(f"{Fore.GREEN}Finished the data configuration process.{Fore.RESET}\n")
                continue
  
            #Pre-Trained Model Inference Environment
            if menu_ch == "3":         
                print(f"{Fore.GREEN}Welcome to the pre-trained model inference environment.{Fore.RESET}")
                print(f"{Fore.RED}Be aware that the weights must match the relevant architecture.{Fore.RESET}")
                model_name = input(f"{Fore.CYAN}Enter the model name that you want to use:{Fore.RESET} \n1. Deep Lab V3+ {Fore.YELLOW}(Default){Fore.RESET} \n2. UNet \n3. PSPNet \n4. Unet++ \n")
                model = load_model(model_name,cfg)
                weight_path = input(f"{Fore.CYAN}Enter the weights path that you want to use:{Fore.RESET} \n")
                model.load_state_dict(torch.load(weight_path))
                pred_image = input(f"{Fore.CYAN}Enter the image path that you want to predict:{Fore.RESET} \n")
                temp_dir, original_size = split_image(pred_image)
                for image in temp_dir:
                    rgb_mask = grey_to_rgb_mask(model,image)                              # Predict the image
                    image = add_mask(image,rgb_mask)
                rebuilt_image_path = rebuild_image(temp_dir, original_size, pred_image)   # Rebuild the image
                print(f"{Fore.GREEN}The prediction has been completed successfully. \nYou can find the prediction in the image path.{Fore.RESET}\n")
                continue
            #Exit
            if menu_ch == "4":                                                            
                print(f"{Fore.GREEN}Thank you for using the ELTA Segmentation Model. Goodbye.{Fore.RESET}")
                exit()
            else:
                print(f"{Fore.RED}Invalid input. Please try again.{Fore.RESET}\n")


            # type_pred = input("visualize (V) or vector map (M)? V/M \n") 




