
#2. initial training script
from ultralytics import YOLO
import os
import torch
from pathlib import Path

def train_model():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("runs/trash_detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load a pretrained YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Using nano version for faster training, can use yolov8x.pt for better accuracy
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data='datasets/trashcan/yolo_format/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        project=str(output_dir),
        name='train',
        save_period=10,
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        fl_gamma=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    print("Training completed!")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    train_model()
