# main.py

"""
Main script for the parking spot detection system.

This script loads the configuration file, reads the input video, and processes the video
to detect parking spots and classify them as either empty or occupied. The results are displayed
on the screen with bounding boxes drawn around parking spots (green for empty, red for occupied).

Instructions:
    1. Ensure that all dependencies listed in requirements.txt are installed.
    2. Set up the project structure as per the instructions in the README.
    3. Provide paths to the input video and mask file in the config.yaml file.
    4. Run the script using the command: `python main.py`

Author: Georgios Athanasiou
Date: 25/09/2024
"""

import os
import cv2
import yaml
from src.video_processor import process_video

# Define the path to the configuration file
config_path = './config.yaml'

def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: The loaded configuration settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

if __name__ == "__main__":
    # Load configuration from file
    config = load_config(config_path)

    # Ensure the video and mask paths are valid
    if not os.path.exists(config['mask_path']):
        raise FileNotFoundError(f"Mask file {config['mask_path']} not found.")
    
    if not os.path.exists(config['video_path']):
        raise FileNotFoundError(f"Video file {config['video_path']} not found.")

    # Run the video processing pipeline
    process_video(config)