from ultralytics import YOLO
import os
import yaml
import logging
from datetime import datetime

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_metrics(metrics, prefix=''):
    """Print training/validation metrics"""
    print(f"\n{prefix}Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    print("-" * 50)

def main():
    print("Starting YOLOv8 training...")
    print(f"Logs will be saved to: training.log")
    
    # Load a pretrained YOLOv8n model
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # You can also use yolov8s.pt, yolov8m.pt, yolov8l.pt, or yolov8x.pt
    
    # Path to the dataset YAML file
    data_yaml = 'yolo_trash_detection/data/dataset.yaml'
    
    # Train the model
    logger.info("Starting training...")
    logger.info(f"Training for 5 epochs with batch size 8 on CPU")
    
    try:
        print(f"Starting training for 5 epochs...")
        results = model.train(
            data=data_yaml,
            epochs=5,   # Reduced to 5 epochs
            imgsz=640,  # Image size for training
            batch=8,    # Reduced batch size for CPU
            device='cpu',  # Use CPU
        workers=4,  # Number of worker threads for data loading
        project='yolo_trash_detection',  # Project name
        name='trash_detection',  # Run name
        save=True,  # Save train checkpoints
        exist_ok=True,  # Overwrite existing project/name
        pretrained=True,  # Use pretrained weights
        optimizer='auto',  # Optimizer to use (SGD, Adam, AdamW, etc.)
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3.0,  # Warmup epochs
        warmup_momentum=0.8,  # Warmup initial momentum
        warmup_bias_lr=0.1,  # Warmup initial bias lr
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # Distribution Focal Loss gain
        hsv_h=0.015,  # Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # Image HSV-Value augmentation (fraction)
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,  # Image scale (+/- gain)
        shear=0.0,  # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # Image flip up-down (probability)
        fliplr=0.5,  # Image flip left-right (probability)
        mosaic=1.0,  # Image mosaic (probability)
        mixup=0.0,  # Image mixup (probability)
        copy_paste=0.0,  # Segment copy-paste (probability)
    )
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    # Evaluate the model on the validation set
    print("\nEvaluating model...")
    metrics = model.val()
    print_metrics(metrics, 'Validation')
    
    # Export the model to ONNX format
    print("\nExporting model to ONNX format...")
    model.export(format='onnx')
    
    print("\nTraining complete!")
    print(f"Model saved to: {os.path.join('yolo_trash_detection', 'trash_detection')}")
    
    # Print final metrics
    if hasattr(model, 'metrics'):
        print_metrics(model.metrics, 'Final')

if __name__ == "__main__":
    main()
