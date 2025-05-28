# Recyclable vs Non-Recyclable Waste Detection

A deep learning-based web application that classifies and detects recyclable and non-recyclable waste items in images using YOLOv8.

![Demo](static/demo.gif)

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Model Training](#model-training)
- [Running the Web App](#running-the-web-app)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [License](#license)

## Features
- 🖼️ Upload images for waste detection
- 🔍 Real-time object detection with bounding boxes
- ♻️ Classifies waste as recyclable or non-recyclable
- 📊 Displays confidence scores for detections
- 🌐 Web-based interface for easy access

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git
- CUDA-compatible GPU (recommended for training)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/recyclablevsnonrecyclable.git
   cd recyclablevsnonrecyclable
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r app_requirements.txt
   ```

## Dataset Setup

### 1. TACO Dataset Setup

The TACO dataset will be automatically downloaded and prepared when you run the training script. 

```bash
# Download and prepare TACO dataset
python prepare_taco_dataset.py
```

### 2. UAVVaste Dataset Setup

For the UAVVaste dataset, follow these steps:

1. **Download the dataset** from [UAVVaste GitHub](https://github.com/smartyfh/DroneWaste) and copy it to the repository folder. 
2. **Preprocess** the dataset (if needed):
   ```bash
   # The model will automatically use the dataset if placed in the correct structure
   # No additional preprocessing is required beyond the initial download and extraction
   ```



## Model Training

### 1. Fine-tune the Trash Detection Model

```bash
# Fine-tune using the pre-trained model
python finetune_trash_detector.py

# Or use the YOLOv8 specific fine-tuning script
python finetune_yolov8.py
```

### 2. Training Configuration

You can customize the training by modifying these parameters in the scripts:

- `epochs`: Number of training epochs (default: 20 in finetune_trash_detector.py, 3 in finetune_yolov8.py)
- `batch`: Batch size (default: 8-16)
- `imgsz`: Input image size (default: 640)
- `device`: 'cuda' for GPU or 'cpu' for CPU training
- `lr0`: Initial learning rate (default: 0.0002-0.01)
- `optimizer`: Optimizer to use (default: 'AdamW' or 'auto')

### 3. Training Outputs

Training outputs will be saved in:
- `runs/trash_detection_finetuned/` (for finetune_trash_detector.py)
- `runs/detect/taco_finetune/` (for finetune_yolov8.py)

### 4. Using the Trained Model

After training, you can use the best model from:
```bash
python trash_detection_app/app.py --model runs/trash_detection_finetuned/finetune/weights/best.pt
```

## Running the Web App

1. **Start the Flask development server**
   ```bash
   python trash_detection_app/app.py
   ```

2. **Access the web interface**
   - Open your browser and go to: `http://localhost:5055`

3. **Upload an image**
   - Click "Choose File" to select an image
   - Click "Detect Trash" to process the image





## Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [UAVVaste Dataset](https://github.com/smartyfh/DroneWaste)
- [TACO Dataset](http://tacodataset.org/) - Trash Annotations in Context for Litter Detection
