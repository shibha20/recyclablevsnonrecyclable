import os
import cv2
from ultralytics import YOLO
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect trash in images or videos using YOLOv8')
    parser.add_argument('--source', type=str, default='0', help='Path to input image, video, or directory (0 for webcam)')
    parser.add_argument('--model', type=str, default='yolo_trash_detection/trash_detection/weights/best.pt', 
                        help='Path to the trained model weights')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--show', action='store_true', default=True, help='Show results')
    args = parser.parse_args()
    
    # Load the trained model
    model = YOLO(args.model)
    
    # Set source
    source = args.source
    if source.isdigit():
        source = int(source)  # Webcam
    
    # Run inference
    results = model(
        source=source,
        conf=args.conf,
        show=args.show,
        save=args.save,
        line_width=2,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
    )
    
    # Print results
    for i, r in enumerate(results):
        if r.boxes is not None:
            print(f"Detected {len(r.boxes)} objects in frame {i+1}")
            for box in r.boxes:
                print(f"  - Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}")

if __name__ == "__main__":
    main()
