# main.py

import sys
from models import ModelA, ModelB  # Import your models here
from utils import utility_function  # Import your utility functions here
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config('src/config/config.yaml')
    
    # Initialize models with configuration parameters
    model_a = ModelA(config['model_a_params'])
    model_b = ModelB(config['model_b_params'])
    
    # Example usage of utility function
    utility_function()

    # Add your main logic here
    print("Models initialized and utility function executed.")

if __name__ == "__main__":
    main()