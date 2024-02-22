import argparse
import image_processing

parser = argparse.ArgumentParser(description='YOLO Video Processing')
parser.add_argument('--cfg', type=str, default='./yolov3.cfg', help='Path to YOLOv3 configuration file')
parser.add_argument('--weights', type=str, default='./yolov3.weights', help='Path to YOLOv3 weights file')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for YOLO')
parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold for YOLO')
parser.add_argument('--circle_diameter_factor', type=float, default=0.3, help='Circle diameter factor for cropping')
parser.add_argument('--camera', type=int, default=0, help='Camera source')

args = parser.parse_args()

image_processing.process_video(args.cfg, args.weights, args.conf, args.nms, args.circle_diameter_factor, args.camera)
