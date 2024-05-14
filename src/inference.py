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

# Function to perform inference on a frame
def inference_frame(model, frame):
    results = model(frame)  # Perform inference
    return results

# Function to draw bounding boxes on a frame
def draw_boxes(frame, results, model):
    for *xyxy, conf, cls in results.xyxy[0]:
        # Draw bounding box
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        # Put class name and confidence
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Main function
def main():
    model_path = 'L:/yolov5_project/models/yolov5s.pt'
    video_path = 'L:/yolov5_project/data/videos/NikolasCage.mp4'
    output_path = 'L:/yolov5_project/output/NikolasCage_new.mp4'

    try:
        # Load YOLOv5 model
        model = load_model(model_path)

        # Open video file
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Process each frame
        for _ in range(num_frames):
            ret, frame = video.read()
            if not ret:
                break

            # Perform inference on frame
            results = inference_frame(model, frame)

            # Draw bounding boxes on frame
            frame_with_boxes = draw_boxes(frame, results, model)

            # Write frame with bounding boxes to output video
            output_video.write(frame_with_boxes)

        # Release video objects
        video.release()
        output_video.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
