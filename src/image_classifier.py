# src/image_classifier.py

"""
Script for training and saving a parking spot image classifier.

This script loads image data of parking spots classified as 'empty' or 'not_empty',
trains a Support Vector Machine (SVM) classifier, evaluates the model's accuracy, and
saves the trained model for use in parking spot detection.

Functions:
    - load_data(input_dir, categories, img_size): Loads and preprocesses image data.
    - train_classifier(x_train, y_train): Trains an SVM classifier using GridSearch for hyperparameter tuning.
    - evaluate_classifier(classifier, x_test, y_test): Evaluates the classifier on the test set.
    - save_model(classifier, model_path): Saves the trained classifier to disk.

Usage:
    Run this script directly to train and save the classifier:
        $ python src/image_classifier.py

Author: Georgios Athanasiou
Date: 25/09/2024
"""

import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Tuple, List

def load_data(input_dir: str, categories: List[str], img_size: Tuple[int, int] = (15, 15)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess image data for classifier training.

    Args:
        input_dir (str): Directory containing categorized images (e.g., 'empty' and 'not_empty').
        categories (List[str]): List of category names corresponding to subfolders in input_dir.
        img_size (Tuple[int, int], optional): Size to which each image will be resized. Default is (15, 15).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Flattened image data and corresponding labels.
    """
    data, labels = [], []
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Category path {category_path} not found.")
        
        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            img = imread(img_path)
            img_resized = resize(img, img_size)
            data.append(img_resized.flatten())
            labels.append(category_idx)

    return np.array(data), np.array(labels)

def train_classifier(x_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Train an SVM classifier using GridSearch for hyperparameter tuning.

    Args:
        x_train (np.ndarray): Training image data.
        y_train (np.ndarray): Training labels.
    
    Returns:
        SVC: Trained classifier with the best hyperparameters.
    """
    classifier = SVC()
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(classifier, parameters)
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_

def evaluate_classifier(classifier: SVC, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the classifier on the test set and print accuracy.

    Args:
        classifier (SVC): Trained classifier.
        x_test (np.ndarray): Test image data.
        y_test (np.ndarray): Test labels.
    """
    y_pred = classifier.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    print(f'{score * 100:.2f}% of samples were correctly classified')

def save_model(classifier: SVC, model_path: str) -> None:
    """
    Save the trained classifier to disk.

    Args:
        classifier (SVC): Trained classifier.
        model_path (str): File path where the model will be saved.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directories if they don't exist
    with open(model_path, 'wb') as model_file:
        pickle.dump(classifier, model_file)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    # Define paths and categories for training
    input_dir = './clf_data'
    categories = ['empty', 'not_empty']
    
    try:
        # Load and preprocess the data
        data, labels = load_data(input_dir, categories)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        # Train the classifier
        classifier = train_classifier(x_train, y_train)

        # Evaluate the classifier
        evaluate_classifier(classifier, x_test, y_test)

        # Save the trained classifier to disk
        save_model(classifier, './models/model.p')

    except Exception as e:
        print(f"Error: {e}")
