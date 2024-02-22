import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def crop_circle(image, center, radius):
    """
    Crops and masks a circular region from an image.

    Parameters:
    image (numpy.ndarray): The input image.
    center (tuple): The center of the circle (x, y).
    radius (int): The radius of the circle.

    Returns:
    numpy.ndarray: The cropped and masked circular region.
    """
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)

    # Masking the area outside the circle
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Creating a bounding box around the circle to crop
    x, y = center
    x1, y1 = max(x - radius, 0), max(y - radius, 0)
    x2, y2 = min(x + radius, image.shape[1]), min(y + radius, image.shape[0])

    # Cropping to the bounding box
    cropped_image = masked_image[y1:y2, x1:x2]

    return cropped_image

def find_dominant_color_hist(image, method='weighted_average'):
    """
    Find the dominant color in an image using color histograms with weighted average.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    tuple: The dominant color in BGR format.
    """
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (image.shape[1] // 2, image.shape[0] // 2), radius, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    h_hist = cv2.calcHist([masked_image], [0], mask, [256], [0, 256]).flatten()
    s_hist = cv2.calcHist([masked_image], [1], mask, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([masked_image], [2], mask, [256], [0, 256]).flatten()

    if method == 'weighted_average':
        h_avg = np.sum(h_hist * np.arange(256)) / np.sum(h_hist)
        s_avg = np.sum(s_hist * np.arange(256)) / np.sum(s_hist)
        v_avg = np.sum(v_hist * np.arange(256)) / np.sum(v_hist)

        dominant_color = (int(h_avg), int(s_avg), int(v_avg))
    else:
        h_peak = np.argmax(h_hist)
        s_peak = np.argmax(s_hist)
        v_peak = np.argmax(v_hist)

        dominant_color = (h_peak, s_peak, v_peak)

    return dominant_color, (h_hist, s_hist, v_hist)

def process_video(cfg_path, weights_path, conf_threshold, nms_threshold, circle_diameter_factor, camera_source):
    
    net = cv2.dnn.readNet("./yolov3.weights",
                        "./yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    confThreshold = 0.5
    nmsThreshold = 0.4
    circle_diameter_factor = 0.3

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes_full = []
        circles = []
        color_circles = []
        confidences = []
        circle_index = 0

        for index, out in enumerate(outs):
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confThreshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes_full.append([x, y, w, h])

                    # Calculate circle params for dominant color extraction
                    radius = int(min(w, h) * circle_diameter_factor / 2)
                    cropped_circle = crop_circle(frame, (center_x, center_y), radius)

                    if cropped_circle.size > 0:
                        dominant_color, hist = find_dominant_color_hist(cropped_circle, method='weighted_average')
                        color_circles.append(dominant_color)
                    else:
                        color_circles.append((0, 0, 0))

                    circles.append((center_x, center_y, radius))
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes_full, confidences, confThreshold, nmsThreshold)

        if isinstance(indices, tuple) and len(indices) > 0:
            indices = indices[0]

        if len(indices) > 0:
            for i in indices:
                i = i[0] if isinstance(i, tuple) or isinstance(i, list) else i
                x, y, w, h = boxes_full[i]
                dominant_color = color_circles[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), dominant_color, 2)
                cx, cy, radius = circles[i]
                cv2.circle(frame, (cx, cy), radius, dominant_color, 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
