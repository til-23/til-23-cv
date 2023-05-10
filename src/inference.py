import torch
import cv2
from PIL import Image
import numpy as np

from reID.test import predict_image
from reID.transforms import Transforms

yolo_model = torch.hub.load('path/to/yolov5', 'custom', path='best.pt', source='local')
reid_model = "reID/model.pth"

suspect = cv2.imread("dummypath")

reid_transforms = Transforms()

def detect_objects(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Perform object detection
    results = yolo_model(img)

    # Print the detected objects with their classes, confidence scores, and bounding box coordinates
    for result in results.xyxy[0].tolist():
        class_id = int(result[5])
        confidence = result[4]
        x1, y1, x2, y2 = result[:4]
        plushie = img[y1:y2, x1:x2]
        match_confidence = float(predict_image(reid_model, suspect, plushie, transform=reid_transforms))
        plushie_class = "suspect" if match_confidence > 0 else "non-suspect"

        print(x1, y1, x2, y2, plushie_class)

image_path = "something"
detect_objects(image_path)



