import yaml
from colorama import Fore, init
init()

def load_yaml(file_path):
    """Load a YAML file."""
    try:
        with open(file_path, 'rt') as f:
            cfg = yaml.safe_load(f.read())
            return cfg
    except FileNotFoundError:
        print(f"Error: YAML file {file_path} not found.")
        return {}

def save_yaml(file_path, data):
    """Save data to a YAML file."""
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Updated YAML file saved at {file_path}.")

def update_cfg(config):
    """Prompt user to update the encoder and pooling settings."""

    # Ask the user for a model name
    while True:
        new_model = input(f"{Fore.CYAN}Enter the number of the model that you want to use: \n1. Deep Lab V3+ (Default) \n2. UNet \n3. PSPNet \n4. Unet++ \n{Fore.RESET}")
        if new_model == "1":
            config['model']['model_name'] = 'DeepLabV3Plus'
            break
        if new_model == "2":
            config['model']['model_name'] = 'UNet'
            break
        if new_model == "3":
            config['model']['model_name'] = 'PSPNet'
            break
        if new_model == "4":
            config['model']['model_name'] = 'UnetPlusPlus'
            break
        else:
            print("Invalid input.")

    # Ask the user for a Loss function name
    while True:
        new_loss = input(f"{Fore.CYAN}Enter the number of the Loss function that you want to use: \n1. Dice Loss (Default) \n2. BCE With Logits Loss \n3. Jaccard Loss \n4. Focal Loss \n{Fore.RESET}")
        if new_loss == "1":
            config['loss']['name'] = 'DiceLoss'
            break
        if new_loss == "2":
            config['loss']['name'] = 'BCEWithLogitsLoss'
            break
        if new_loss == "3":
            config['loss']['name'] = 'JaccardLoss'
            break
        if new_loss == "4":
            config['loss']['name'] = 'FocalLoss'
            break
        else:
            print("Invalid input.")

    # Ask the user for an optimizer name
    while True:
        new_optimizer = input(f"{Fore.CYAN}Enter the number of the optimizer that you want to use: \n1. AdamW (Default) \n2. Adam \n3. SGD \n{Fore.RESET}")
        if new_optimizer == "1":
            config['train']['optimizer_name'] = 'AdamW'
            break
        if new_optimizer == "2":
            config['train']['optimizer_name'] = 'Adam'
            break
        if new_optimizer == "3":
            config['train']['optimizer_name'] = 'SGD'
            break
        else:
            print("Invalid input.")

    # Ask the user for a Weight Decay value
    while True:
        new_weight_decay = input(f"{Fore.CYAN}Enter the weight decay value that you want to use: \n1. 0.0001 (Default) \n2. 0.00001 \n3. 0.000001 \n{Fore.RESET}")
        if new_weight_decay == "1":
            config['train']['weight_decay'] = 0.0001
            break
        if new_weight_decay == "2":
            config['train']['weight_decay'] = 0.00001
            break
        if new_weight_decay == "3":
            config['train']['weight_decay'] = 0.000001
            break
        else:
            print("Invalid input.")

    return config




if __name__ == "__main__":
    yaml_file = "config.yaml"

    # Load the YAML configuration
    config = load_yaml(yaml_file)

    if config:
        # Update the encoder_name and pooling values
        config = update_encoder_and_pooling(config)

        # Save the updated YAML configuration
        save_yaml(yaml_file, config)
    else:
        print("Failed to load configuration.")