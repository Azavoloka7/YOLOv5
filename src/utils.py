import torch

# Function to load YOLOv5 model
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    return model

# Function to perform inference on an image
def inference_image(model, image_path):
    img = cv2.imread(image_path)  # Load image with OpenCV
    results = model(img)           # Perform inference
    return results

# Function to draw bounding boxes on an image
def draw_boxes(image, results):
    for *xyxy, conf, cls in results.xyxy[0]:
        # Draw bounding box
        cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        # Put class name and confidence
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
