from colorama import Fore, init
init()
from train import train
from testEvaluation import test_evaluation
from utils import *
import yaml
from utils.cfg_utils import *

##################################################################################

if __name__ == "__main__":

    yaml_file = '/workspace/config.yaml'
    cfg = load_yaml(yaml_file)
    if cfg:
        print(f"{Fore.LIGHTBLUE_EX}Welcome to the segmentation model{Fore.RESET}")
        default_cfg = input(f"{Fore.CYAN}Do you want to use the default configuration? Y/n \n{Fore.RESET}")
        if default_cfg == "Y":
            print(f"{Fore.CYAN}Using default configuration..{Fore.RESET}")
        if default_cfg == "n":
            cfg = update_cfg(cfg)


    want_to_train = input('want to run a train? Y/n \n')
    if want_to_train == "Y":

        train(cfg) #Start the train

        best_model_dir = test_evaluation(train_dir,model,cfg) #check which version is the best
    

            # type_pred = input("visualize (V) or vector map (M)? V/M \n") 
            # if type_pred == "V":
            #     pred_pic(model,version_model_dir,path)
            # if type_pred == "M":
            #     pred_map(model,version_model_dir,path)
            # ctn = input("want to continue? Y/n : \n")
            # if ctn != "Y":
            #     break



