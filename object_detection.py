import cv2
import numpy as np

class ObjectDetector:
    """
    A class for object detection using a pre-trained model.

    Attributes:
        net (cv2.dnn_Net): The loaded DNN model.
        output_layers (list): List of names of the output layers of the model.
        confThreshold (float): Threshold for filtering out weak detections based on confidence scores.
        nmsThreshold (float): Threshold for non-maximum suppression.

    Methods:
        detect_objects(frame): Detects objects in the given frame.
    """
    def __init__(self, weights_path, config_path, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initializes the ObjectDetector with the model weights, configuration, and thresholds.

        Parameters:
            weights_path (str): Path to the model weights file.
            config_path (str): Path to the model configuration file.
            confidence_threshold (float, optional): Confidence threshold to filter detections. Defaults to 0.5.
            nms_threshold (float, optional): Threshold for non-maximum suppression. Defaults to 0.4.
        """
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            print("CUDA not available - defaulting to CPU. Performance may be significantly reduced.")

        layer_names = self.net.getLayerNames()
        unconnected_out_layers = self.net.getUnconnectedOutLayers()
        if len(unconnected_out_layers.shape) == 2:
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:
            self.output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
        self.confThreshold = confidence_threshold
        self.nmsThreshold = nms_threshold

    def detect_objects(self, frame):
        """
        Detects objects in the given frame using the loaded DNN model.

        Parameters:
            frame (numpy.ndarray): The frame in which objects are to be detected.

        Returns:
            tuple: A tuple containing three lists:
                - nms_boxes (list): The bounding boxes of detected objects after applying NMS.
                - nms_confidences (list): The confidence scores of detected objects after applying NMS.
                - nms_class_ids (list): The class IDs of detected objects after applying NMS.
        """
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        nms_boxes = [boxes[i] for i in indices.flatten()]
        nms_confidences = [confidences[i] for i in indices.flatten()]
        nms_class_ids = [class_ids[i] for i in indices.flatten()]

        return nms_boxes, nms_confidences, nms_class_ids
