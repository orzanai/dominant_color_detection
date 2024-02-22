
# Introduction

This project focuses on color detection using OpenCV, a powerful computer vision library. The core functionality of this application is to detect dominant colors in real-time through a webcam feed. We utilize advanced image processing techniques and object detection algorithms to achieve this goal.




## Approach

My approach integrates several key steps:

- Object Detection: Using YOLOv3, a state-of-the-art object detection algorithm, we identify objects in each frame of the webcam feed.
- Color Detection: For each detected object, we extract a circular region and compute its dominant color. This is achieved through color histograms and weighted averages.
- Real-Time Processing: The application operates in real-time, updating the detected objects and their colors in each frame.

## Tech Stack

| Tech          | Purpose                                                               |
| ----------------- | ------------------------------------------------------------------ |
| Python | The primary programming language used for this project. |
| OpenCV | For computer vision tasks, used for both object detection and color analysis. |
| YOLO | An object detection algorithm known for its accuracy and efficiency.|
| NumPy | Utilized for numerical operations on image matrices.|


## Methodology

The code follows a specific sequence of operations:

- Capture the webcam feed frame by frame.
- Apply YOLOv3 to detect objects in each frame.
- For each detected object, calculate a circular region and extract its dominant color.
- Display the results in real-time, showing the detected objects bounded by rectangles colored based on the dominant color in the corresponding region.

