# Parking Spot Detection System

This project implements a system that analyzes video footage of parking lots to detect whether parking spots are empty or occupied. It utilizes OpenCV for video processing, scikit-learn for the classification model, and scikit-image for image handling.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Prepare Data](#2-prepare-data)
  - [3. Train the Classifier](#3-train-the-classifier)
  - [4. Run the Parking Spot Detection](#4-run-the-parking-spot-detection)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
  - [Classifier Training](#classifier-training)
  - [Video Processing](#video-processing)
- [Contributing](#contributing)
- [License](#license)

## Features
- Detects parking spots in a video based on a mask image.
- Classifies parking spots as either empty or occupied using a pre-trained Support Vector Machine (SVM) classifier.
- Displays real-time video with bounding boxes around parking spots (green for empty, red for occupied).
- Easily configurable via a `config.yaml` file.

## Project Structure

## Requirements

To get started, ensure that you have Python 3.7+ installed and install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt

### Core Dependencies:
- **OpenCV**: For video processing and image manipulation.
- **scikit-learn**: For training and running the SVM classifier.
- **scikit-image**: For handling image transformations.
- **numpy**: For numerical computations and array manipulations.
- **PyYAML**: For reading the configuration file (`config.yaml`).
```

## Setup

### 1. Install Dependencies
Run the following command to install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure that your data is correctly structured in the `data/` and `clf_data/` directories.

1. **Mask and Video Files**: Place the mask image and video file in the `data/` directory. The mask is used to identify the parking spots in the video, and the video contains the footage to be processed.


2. **Classifier Training Data**: Organize the training images for the classifier in the `clf_data/` directory. Use two subdirectories:
   - `empty/`: Contains images of empty parking spots.
   - `not_empty/`: Contains images of occupied parking spots.

### 3. Train the Classifier

To train the SVM classifier that distinguishes between empty and occupied parking spots, run the following script:

```bash
python3 src/image_classifier.py
``` 

This script performs the following steps:

- **Load Data**: Reads images from the `clf_data/` directory. The `empty/` folder contains images of empty spots, while the `not_empty` folder contains images of occupied spots.
- **Preprocess Data**: Resizes the images to a standard size (15x15 pixels) and flattens them for input into the model.
- **Train the Classifier**: Trains a Support Vector Machine (SVM) model using the preprocessed images, applying grid search for hyperparameter optimization.
- **Evaluate the Model**: Splits the data into training and testing sets, then evaluates the model's accuracy on the test set.
- **Save the Model**: Saves the trained model to `models/model.p` for use in the parking spot detection system.

### 4. Run the Parking Spot Detection

Once the classifier is trained (or if you already have a trained model saved in `models/model.p`), you can run the parking spot detection on your video by executing the following command:

```bash
python3 main.py
```

This script performs the following steps:

- **Load the Video and Mask**: Reads the video file from `data/parking_crop_loop.mp4` and the mask image from `data/mask_crop.png`.
- **Detect Parking Spots**: Uses the mask to identify the locations of parking spots in the video through connected component analysis.
- **Classify Parking Spots**: For each detected parking spot, the pre-trained SVM model classifies the spot as either empty

## Configuration

The project uses a `config.yaml` file to configure paths and parameters. Make sure the file has the correct paths to your data and any additional settings:

```yaml
mask_path: './data/mask_crop.png'           # Path to the mask image
video_path: './data/parking_crop_loop.mp4'  # Path to the input video
frame_step: 30                              # Number of frames to skip between processing
```

## Technical Details

### Classifier Training

The classifier is a Support Vector Machine (SVM) model trained to distinguish between empty and occupied parking spots. Images are resized to 15x15 pixels and flattened before being used for training. GridSearchCV is utilized for hyperparameter tuning to find the best parameters for the model.

- **Training Script**: `src/image_classifier.py`
- **Model File**: `models/model.p`

### Video Processing

The video processing pipeline follows these steps:
1. **Video and Mask Loading**: Loads the input video from `data/parking_crop_loop.mp4` and the mask image from `data/mask_crop.png`.
2. **Parking Spot Detection**: Detects parking spots in the video using connected component analysis based on the mask, which provides the location of each parking spot.
3. **Spot Classification**: For each detected parking spot, the pre-trained SVM model classifies the spot as either empty or occupied.
4. **Bounding Box Drawing**: Draws bounding boxes around each parking spot. Green boxes indicate empty spots, and red boxes indicate occupied spots.
5. **Real-time Display**: The processed video is displayed in real-time with bounding boxes overlaid. You can press **'q'** to exit the display.

- **Video Processing Script**: `src/video_processor.py`

## Contributing

Contributions are welcome! To contribute to this project, follow these steps:

1. **Fork the Repository**: Click the "Fork" button at the top of the repository page on GitHub to create your own copy of the repository.
2. **Clone the Repository**: Clone your forked repository to your local machine.
   ```bash
   git clone https://github.com/ga83wuw/parkingVision.git
   ```
3. **Create a New Branch**: Create a new branch for your feature or bug fix.
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Your Changes**: Implement your feature or bug fix in your local branch.

5. **Commit Your Changes**: Once your changes are complete, commit them with a clear and descriptive commit message.
   ```bash
   git commit -m "Add: description of the feature or fix"
   ```
6. **Push Your Changes**: Push your local branch to your forked repository on GitHub.
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**: 
   Go to your forked repository on GitHub, click the "New Pull Request" button, and select your feature branch. Submit the Pull Request (PR) to the original repository's `main` branch. In the PR description, provide a summary of the changes and explain why they are necessary.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the software, as long as you include the original license file. 

For more details, refer to the [LICENSE](./LICENSE) file.
