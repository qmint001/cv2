import numpy as np
import cv2
import time
import sys
import os
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics import YOLO

# Constants
INPUT_WIDTH = 640  # Width of network's input image, larger is slower but more accurate
INPUT_HEIGHT = 640  # Height of network's input image, larger is slower but more accurate
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold, high values filter out low confidence detections
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold, higher values result in duplicate boxes per object

# Text parameters
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 2

# Load the class names from the COCO dataset
CLASSES = yaml_load(check_yaml("coco8.yaml"))["names"]

# Generate a color palette for the classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Function to Draw the Detections
def draw_detections(img, box, score, class_id):
    x1, y1, w, h = box
    color = COLORS[class_id]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    label = f"{CLASSES[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), FONT_FACE, FONT_SCALE, (0, 0, 0), THICKNESS, cv2.LINE_AA)

# Load the YOLOv8 model with OpenCV-DNN and do the Forword Pass
def inference(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    image_data = net.forward(output_layers)
    return image_data

# Function to do the post processing with the detections from our model
def postprocess(input_image, output):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    img_height, img_width = input_image.shape[:2]
    x_factor = img_width / INPUT_WIDTH
    y_factor = img_height / INPUT_HEIGHT

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= CONFIDENCE_THRESHOLD:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        draw_detections(input_image, box, score, class_id)

    return input_image

# Function to put the Inference Time and FPS
def put_efficiency(input_img, inference_time, is_video=False):
    if is_video:
        fps = 1 / inference_time
        label = "FPS: %.2f" % fps
    else:
        label = "Inference Time: %.2f ms" % (inference_time * 1000.0)
    # print(label)
    cv2.putText(input_img, label, (10, 50), FONT_FACE, FONT_SCALE, (0, 255, 0), THICKNESS, cv2.LINE_AA)

# Run inference for an Image
def process_image(image_path, net):
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    start_time = time.time()
    detections = inference(frame, net)
    img = postprocess(frame.copy(), detections)
    inference_time = time.time() - start_time
    put_efficiency(img, inference_time, is_video=False)
    cv2.imshow("YOLOv8 Object Detection", img)
    cv2.imwrite("output.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run Inference with a video
def process_video(video_path, net, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        start_time = time.time()
        detections = inference(frame, net)
        inference_time = time.time() - start_time
        img = postprocess(frame.copy(), detections)
        put_efficiency(img, inference_time, is_video=True)
        out.write(img)
        # cv2.imshow('YOLOv8 Object Detection', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# to run the inference from terminal 
def main(input_path, is_video=False, output_path=None):
    model = YOLO("yolov8s.pt") # change according to your need between YOLOv8 and YOLOv9 both will works
    model.export(format="onnx")
    net = cv2.dnn.readNetFromONNX("yolov8s.onnx")

    if is_video:
        process_video(input_path, net, output_path)
    else:
        process_image(input_path, net)


if __name__ == "__main__":
    input_path = sys.argv[1]
    is_video = sys.argv[2].lower() == "video"
    output_path = sys.argv[3] if is_video else None

    main(input_path, is_video, output_path)

# for image inference - command your treminal:
# python yolo_inference.py /path/to/image.jpg image

# for video inference - command your treminal:
# python yolo_inference.py /path/to/video.mp4 video /path/to/output_video.mp4