# src/video_processor.py

"""
Module for processing the parking lot video to detect and classify parking spots.

This module reads a video file and a mask, detects parking spots using connected component analysis,
and classifies each spot as either empty or occupied using a pre-trained classifier. The result is
displayed in real-time with bounding boxes drawn around the parking spots.

Green box: Empty parking spot.
Red box: Occupied parking spot.

Functions:
    - process_video(config): Process the video file based on the provided configuration.

Author: Georgios Athanasiou
Date: 25/09/2024
"""

import cv2
from src.utils import get_parking_spots_bboxes, empty_or_not

def process_video(config: dict) -> None:
    """
    Process the video to detect and classify parking spots.
    
    This function loads the mask image and the video, identifies the parking spots using 
    connected components, and determines whether each parking spot is empty or not using 
    the pre-trained classifier. The results are shown with bounding boxes drawn over the video.

    Args:
        config (dict): A dictionary containing configuration settings, including:
            - mask_path (str): Path to the mask image.
            - video_path (str): Path to the video file.
            - frame_step (int): Number of frames to skip between processing steps.
    
    Raises:
        FileNotFoundError: If the mask or video file does not exist.
    """
    
    # Load the mask image
    mask = cv2.imread(config['mask_path'], 0)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found at {config['mask_path']}")

    # Load the video
    video_path = config['video_path']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found or could not be opened: {video_path}")

    # Extract parking spots using connected component analysis
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)

    # Initialize the status of each parking spot
    spots_status = [None for _ in spots]

    frame_nmr = 0
    step = config['frame_step']
    ret = True

    while ret:
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if the video ends

        # Process every Nth frame (determined by frame_step)
        if frame_nmr % step == 0:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                # Extract the region of interest (ROI) corresponding to the parking spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w]
                # Classify the parking spot as empty or occupied
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_indx] = spot_status

        # Draw bounding boxes around each parking spot
        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spot
            # Green for empty, red for occupied
            color = (0, 255, 0) if spot_status else (0, 0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        # Display the processed frame with bounding boxes
        cv2.imshow('Parking Spot Detection', frame)
        
        # Exit if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_nmr += 1

    # Release the video capture and close display windows
    cap.release()
    cv2.destroyAllWindows()