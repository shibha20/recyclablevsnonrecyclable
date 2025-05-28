# Recyclable vs Non-Recyclable Waste Detection

A deep learning-based web application that detects trash in images using YOLOv8.

![Demo](static/demo.gif)



## Features
- 🖼️ Upload images for waste detection
- 🔍 Real-time trash detection with bounding boxes
- 📊 Displays confidence scores for detections
- 🌐 Web-based interface for easy access

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/trashdetectionusingyolo.git
   cd recyclablevsnonrecyclable
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r app_requirements.txt
   ```

## Dataset Setup

### 1. TACO Dataset Setup

The TACO dataset will be automatically downloaded and prepared when you run the below script. 

```bash
# Download and prepare TACO dataset
python prepare_taco_dataset.py
```

### 2. UAVVaste Dataset Setup

For the UAVVaste dataset, follow these steps:

1. **Download the dataset** from [UAVVaste GitHub](https://github.com/PUTvision/UAVVaste/tree/main) and copy it to the repository folder. 
2. **Preprocess** the dataset using the script below:
   convert_coco_to_yolo.py
   ```



## Model Training

Base Model: YOLOv8n (pre-trained on COCO) 
First Training: TACO dataset (ground-level trash) - train_trash_detector.py 
Fine-tuning: UAVVaste (aerial trash) - finetune_trash_detector.py


### 4. Using the Trained Model

After training, you can use the  model from:
python trash_detection_app/app.py

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
- [UAVVaste Dataset]([https://github.com/smartyfh/DroneWaste](https://github.com/PUTvision/UAVVaste/tree/main)))
- [TACO Dataset](http://tacodataset.org/) 
