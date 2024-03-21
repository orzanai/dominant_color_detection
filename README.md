
# Introduction

This project focuses on color detection using OpenCV, a powerful computer vision library. The core functionality of this application is to detect dominant colors in real-time through a webcam feed. I utilize advanced image processing techniques and object detection algorithms to achieve this goal.


## Screencap

![color-detection-cars](https://github.com/orzanai/color_detection/blob/main/test.gif)


## Approach

My approach integrates several key steps:

- Object Detection: Using YOLO, a state-of-the-art object detection algorithm, I identify objects in each frame of the webcam feed.
- Color Detection: For each detected object, I extract a circular region and thenperforming k-means clustering on the pixels within this region to find clusters of similar colors. The dominant color is calculated as a weighted average of these clusters, where the weight is the proportion of pixels in each cluster
- Real-Time Processing: The application operates in real-time, updating the detected objects and their colors in each frame.

## Tech Stack

| Tech          | Purpose                                                               |
| ----------------- | ------------------------------------------------------------------ |
| Python | The primary programming language used for this project. |
| OpenCV | For computer vision tasks, used for both object detection and color analysis. |
| YOLO | An object detection algorithm known for its accuracy and efficiency.|
| NumPy | Utilized for numerical operations on image matrices.|
| Scikit-learn | Used for performing k-means clustering.|


## Methodology

The code follows a specific sequence of operations:

- Capture the webcam feed frame by frame.
- Apply YOLO with OpenCV to detect objects in each frame.
- For each detected object, calculate a circular region and extract its dominant color.
- Display the results in real-time, showing the detected objects bounded by rectangles colored based on the dominant color in the corresponding region.


## Run Locally

After changing paths for YOLO weights and configuration files in main.py you can run project with

```bash
  python main.py

```
