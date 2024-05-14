import torch
import numpy as np
import cv2
import os

# Function to load YOLOv5 model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    # Load the YOLOv5 model directly
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    return model

# Function to perform inference on an image
def inference_image(model, image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    img = cv2.imread(image_path)  # Load image with OpenCV
    results = model(img)           # Perform inference
    return results

# Function to draw bounding boxes on an image
def draw_boxes(image, results, model):
    for *xyxy, conf, cls in results.xyxy[0]:
        # Draw bounding box
        cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        # Put class name and confidence
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Function to save annotated image with bounding boxes
def save_annotated_image(image, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved: {output_path}")

# Main function
def main():
    model_path = 'L:/yolov5_project/models/yolov5s.pt'
    image_path = 'L:/yolov5_project/data/images/example.jpg'
    output_dir = 'L:/yolov5_project/output'

    try:
        # Load YOLOv5 model
        model = load_model(model_path)

        # Perform inference on image
        results = inference_image(model, image_path)

        # Draw bounding boxes on image
        image = cv2.imread(image_path)
        image_with_boxes = draw_boxes(image, results, model)

        # Save annotated image with bounding boxes
        image_name = os.path.basename(image_path)
        save_annotated_image(image_with_boxes, output_dir, image_name)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
