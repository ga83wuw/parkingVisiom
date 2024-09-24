# src/utils.py

"""
Utility functions for parking spot detection and classification.

This module contains helper functions to:
1. Classify parking spots as empty or occupied using a pre-trained classifier.
2. Extract bounding boxes for parking spots from connected component analysis of the parking mask.

Functions:
    - empty_or_not(spot_bgr): Classifies a parking spot as empty or not.
    - get_parking_spots_bboxes(connected_components): Extracts bounding boxes for parking spots.

Author: Georgios Athanasiou
Date: 25/09/2024
"""

import pickle
import numpy as np
import cv2
from skimage.transform import resize
from typing import List, Tuple

# Constants for parking spot classification
EMPTY = True
NOT_EMPTY = False

# Load pre-trained classifier model
try:
    MODEL = pickle.load(open("models/model.p", "rb"))
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Ensure 'models/model.p' exists.")

def empty_or_not(spot_bgr: np.ndarray) -> bool:
    """
    Classify a parking spot as empty or occupied using a pre-trained model.

    Args:
        spot_bgr (np.ndarray): The BGR image (NumPy array) of the parking spot to classify.
    
    Returns:
        bool: True if the spot is classified as empty, False if classified as occupied.
    
    Raises:
        ValueError: If the input image is not of the expected shape.
    """
    # Resize the image to the expected input size for the classifier
    if spot_bgr.shape[0] == 0 or spot_bgr.shape[1] == 0:
        raise ValueError("Input image has invalid dimensions.")

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data = np.array([img_resized.flatten()])

    # Predict whether the spot is empty or occupied
    y_output = MODEL.predict(flat_data)

    return EMPTY if y_output == 0 else NOT_EMPTY

def get_parking_spots_bboxes(connected_components) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes for each parking spot from connected component analysis.

    Args:
        connected_components: Result of cv2.connectedComponentsWithStats containing label, size, and bounding box info.
    
    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding boxes (x, y, width, height) for each parking spot.
    """
    total_labels, label_ids, values, _ = connected_components
    slots = []

    # Iterate through each label (ignoring the first, which is the background)
    for i in range(1, total_labels):
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])

        # Append the bounding box for each parking spot
        slots.append((x1, y1, w, h))

    return slots