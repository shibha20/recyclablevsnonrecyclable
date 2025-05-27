import os
import shutil
import zipfile
import requests
from pathlib import Path
import yaml
from tqdm import tqdm
import subprocess
import json
import pandas as pd
from urllib.parse import urlparse

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Failed to download the file completely")
        return False
    return True

def download_image(url, save_path):
    """Download an image from URL with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get the file extension from URL or default to jpg
            parsed_url = urlparse(url)
            file_ext = os.path.splitext(parsed_url.path)[1].lower()
            if not file_ext or len(file_ext) > 5:  # If no extension or too long
                file_ext = '.jpg'
                
            # Ensure the save path has the correct extension
            save_path = str(save_path) + file_ext
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return save_path
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download {url} after {max_retries} attempts: {e}")
                return None
            continue

def prepare_taco_dataset():
    # Create directories
    dataset_dir = Path("datasets/taco")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotations_path = dataset_dir / "data" / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations JSON not found at {annotations_path}")
    
    print("Loading annotations...")
    with open(annotations_path) as f:
        data = json.load(f)
    
    # Create a mapping from image ID to image info
    image_info = {img['id']: img for img in data['images']}
    print(f"Found {len(image_info)} images in annotations")
    
    # Create a mapping from image ID to filename
    img_id_to_filename = {}
    
    # Find all image files in the data directory
    data_dir = dataset_dir / "data"
    print("Scanning for existing image files...")
    
    # Look for common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Create a mapping from image ID to file path
    for img_id, img_info in tqdm(image_info.items(), desc="Mapping existing images"):
        # Try to find the image file with any common extension
        for ext in image_extensions:
            # Check both the original path and the batch structure
            possible_paths = [
                data_dir / img_info['file_name'],
                data_dir / f"batch_{img_id//100:02d}" / f"{img_id:06d}{ext}",
                data_dir / f"{img_id}{ext}",
                data_dir / f"{img_id:06d}{ext}",
            ]
            
            for path in possible_paths:
                if path.exists():
                    img_id_to_filename[img_id] = path.relative_to(data_dir)
                    break
            else:
                continue
            break
        else:
            print(f"Warning: Could not find image file for ID {img_id}")
    
    print(f"Found {len(img_id_to_filename)} existing image files")
    
    # Install required packages
    print("Installing required packages...")
    subprocess.run(["pip", "install", "-r", str(dataset_dir / "requirements.txt")])
    
    # Create YOLO directory structure
    yolo_dir = dataset_dir / "yolo_format"
    (yolo_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images/test").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels/val").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels/test").mkdir(parents=True, exist_ok=True)
    
    # Convert TACO annotations to YOLO format
    print("Converting annotations to YOLO format...")
    
    # We already loaded the annotations earlier
    
    # Create train/val/test split (80/10/10)
    n_images = len(data['images'])
    train_end = int(n_images * 0.8)
    val_end = train_end + int(n_images * 0.1)
    
    # Process images and annotations
    print("Processing images and annotations...")
    for i, img_info in enumerate(tqdm(data['images'], desc="Processing images")):
        img_id = img_info['id']
        
        # Get the downloaded image path from our mapping
        if img_id not in img_id_to_filename:
            print(f"Warning: No downloaded image found for ID {img_id}")
            continue
            
        img_relative_path = img_id_to_filename[img_id]
        img_path = dataset_dir / 'data' / img_relative_path
        
        # Update the filename in the image info to match our downloaded file
        img_info['file_name'] = str(img_relative_path)
        
        # Determine split based on index
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'
        
        # Skip if image doesn't exist
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
            
        # Determine split
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'
        
        # Create destination directory if it doesn't exist
        dest_img_dir = yolo_dir / 'images' / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir = yolo_dir / 'labels' / split
        dest_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image to YOLO directory
        dest_img_path = dest_img_dir / f"{img_id}.jpg"
        if not dest_img_path.exists():
            shutil.copy(img_path, dest_img_path)
        
        # Create YOLO format annotations
        annotations = [a for a in data['annotations'] if a['image_id'] == img_id]
        if annotations:
            with open(dest_label_dir / f"{img_id}.txt", 'w') as f:
                for ann in annotations:
                    # Convert COCO bbox [x,y,width,height] to YOLO [x_center, y_center, width, height] (normalized)
                    x, y, w, h = ann['bbox']
                    img_w, img_h = img_info['width'], img_info['height']
                    
                    # Skip invalid annotations
                    if w <= 0 or h <= 0 or x >= img_w or y >= img_h:
                        continue
                        
                    # Ensure bbox is within image bounds
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(img_w - 1, x + w)
                    y2 = min(img_h - 1, y + h)
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    x_center = (x1 + w/2) / img_w
                    y_center = (y1 + h/2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    
                    # Ensure values are within [0,1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    # Write YOLO format (class_id x_center y_center width height)
                    # Using class 0 for all trash objects
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    # Create YOLO dataset YAML
    data_yaml = {
        'path': str(yolo_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'trash',
        },
        'nc': 1
    }
    
    with open(yolo_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print("\nDataset prepared successfully!")
    print(f"Dataset location: {yolo_dir.absolute()}")
    print(f"Training images: {len(list((yolo_dir / 'images/train').glob('*.jpg')))}")
    print(f"Validation images: {len(list((yolo_dir / 'images/val').glob('*.jpg')))}")
    print(f"Test images: {len(list((yolo_dir / 'images/test').glob('*.jpg')))}")

if __name__ == "__main__":
    prepare_taco_dataset()
