from ultralytics import YOLO
import torch
from pathlib import Path

def finetune_model():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    model_path = 'yolo_trash_detection/trash_detection/weights/best.pt'
    dataset_yaml = 'datasets/trashcan/yolo_format/dataset.yaml'
    output_dir = Path("runs/trash_detection_finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load the existing model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Set model to training mode
    model.train()
    
    # Fine-tuning parameters - optimized for quick fine-tuning
    print("Starting fine-tuning...")
    results = model.train(
        data=dataset_yaml,
        epochs=20,  # Reduced epochs for quicker fine-tuning
        imgsz=640,
        batch=16,  # Increased batch size for faster training
        device=device,
        workers=4,
        project=str(output_dir),
        name='finetune',
        save_period=10,
        pretrained=False,  # Use our existing weights
        optimizer='AdamW',  # Better for fine-tuning
        lr0=0.0002,  # Slightly higher learning rate for faster convergence
        lrf=0.02,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=1.0,  # Shorter warmup
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
        copy_paste=0.0,
        freeze=15  # Freeze more layers (15) to preserve more of the original model
    )
    
    print("Fine-tuning completed!")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    finetune_model()
