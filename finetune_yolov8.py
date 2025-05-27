import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def main():
    # Paths
    base_dir = Path("datasets/taco/yolo_format")
    model_path = "yolo_trash_detection/trash_detection/weights/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Dataset config
    data_yaml = base_dir / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found at {data_yaml}")
    
    # Training arguments
    train_args = {
        'data': str(data_yaml),  # Path to data.yaml
        'epochs': 3,             # Number of training epochs (reduced for quick testing)
        'batch': 8,              # Batch size
        'imgsz': 640,            # Image size
        'device': 'cpu',         # Use CPU for training
        'workers': 4,            # Number of worker threads
        'project': 'runs/detect',# Save directory
        'name': 'taco_finetune', # Run name
        'exist_ok': True,        # Overwrite existing runs
        'pretrained': True,      # Use pre-trained weights
        'optimizer': 'auto',     # Auto-select optimizer
        'lr0': 0.01,            # Initial learning rate
        'lrf': 0.01,            # Final learning rate (lr0 * lrf)
        'momentum': 0.937,       # SGD momentum
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3.0,    # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,   # Warmup bias learning rate
        'box': 0.05,             # Box loss gain
        'cls': 0.5,              # Class loss gain
        'fliplr': 0.5,           # Image flip left-right (probability)
        'mosaic': 1.0,           # Image mosaic (probability)
    }
    
    # Start fine-tuning
    print("Starting fine-tuning...")
    try:
        # Train the model with the specified arguments
        results = model.train(**train_args)
        
        # Save the fine-tuned model
        model.save('yolo_trash_detection/trash_detection/weights/finetuned.pt')
        print("\nFine-tuning completed successfully!")
        print(f"Model saved to: yolo_trash_detection/trash_detection/weights/finetuned.pt")
        
        # Print training results
        print("\nTraining Results:")
        print(f"- mAP@0.5: {results.metrics.map50:.4f}")
        print(f"- mAP@0.5:0.95: {results.metrics.map:.4f}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Please check the error message and adjust the training parameters if needed.")

if __name__ == "__main__":
    main()
