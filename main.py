import cv2
from object_detection import ObjectDetector
from color_detection import find_combined_dominant_color, get_dynamic_circle_diameter, crop_circle

object_detector = ObjectDetector("yolo.weights_path",
                                 "yolov.cfg_path")

cap = cv2.VideoCapture(0) # 0 for real-time video capture, or path for video processing.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    color_circles = [] # Stores the dominant color of each detected circle, (R, G, B)
    boxes, confidences, class_ids = object_detector.detect_objects(frame)

    for i,box in enumerate(boxes):
        x, y, w, h = box
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        radius = get_dynamic_circle_diameter(w, h)
        cropped_circle = crop_circle(frame, (center_x, center_y), radius)
        if cropped_circle.size > 0:
            dominant_color = find_combined_dominant_color(cropped_circle, n_clusters=3)
            color_circles.append(dominant_color)
        else:
            color_circles.append((0,0,0))

        dominant_color = color_circles[i]
        cv2.rectangle(frame, (x,y), (x+w, y+h), dominant_color, 2)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
