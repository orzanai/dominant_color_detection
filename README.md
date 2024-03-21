
# Introduction

This project focuses on color detection using OpenCV, a powerful computer vision library. The core functionality of this application is to detect dominant colors in real-time through a webcam feed. I utilize advanced image processing techniques and object detection algorithms to achieve this goal.


## Screencap

![color-detection-cars](https://github.com/orzanai/color_detection/blob/main/test.gif)


## Approach

My approach integrates several key steps:

- Object Detection: Using YOLO, a state-of-the-art object detection algorithm, I identify objects in each frame of the webcam feed.
- Color Detection: For each detected object, I extract a dynamic circular region and then performed k-means clustering k-means clustering to group similar colors and selects the dominant color based on cluster compactness and coverage.
- Real-Time Processing: The application operates in real-time, updating the detected objects and their colors in each frame.


## Color Detection Process

- The image is first converted into a 2D array of pixels, where each pixel is represented by its color in the BGR (Blue, Green, Red) color space.
- K-means clustering is applied to this array with a specified number of clusters (n_clusters). Each cluster represents a group of pixels with similar colors.
- For each cluster, two metrics are calculated:
  - Coverage: The proportion of pixels in the image that belong to each cluster.
  - Compactness: A measure of how close the pixels within a cluster are to the cluster's center. It's calculated as the average distance from each pixel in the cluster to the cluster's centroid.
- Each cluster is scored based on a combination of its coverage and compactness. This scoring aims to favor clusters that are both large (high coverage) and cohesive (high compactness).
- The cluster with the highest score is selected, and its centroid color is chosen as the dominant color.

## Tech Stack

| Tech          | Purpose                                                               |
| ----------------- | ------------------------------------------------------------------ |
| Python | The primary programming language used for this project. |
| OpenCV | For computer vision tasks, used for both object detection and color analysis. |
| YOLO | An object detection algorithm known for its accuracy and efficiency.|
| NumPy | Utilized for numerical operations on image matrices.|


## Run Locally

After changing paths for YOLO weights and configuration files in main.py you can run project with

```bash
  python main.py

```
