#1.Convert COCO format to YOLO format
import json
import os
import cv2
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO format to YOLO format
    """
    # COCO format: [x_min, y_min, width, height] in absolute coordinates
    # YOLO format: [x_center, y_center, width, height] in normalized coordinates (0-1)
    
    x_min, y_min, width, height = bbox
    
    # Calculate center coordinates
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    
    # Normalize width and height
    width = width / img_width
    height = height / img_height
    
    return [x_center, y_center, width, height]

def main():
    # Paths
    coco_json_path = 'UAVVasteDataset/annotations/annotations.json'
    images_dir = 'UAVVasteDataset/images'
    output_dir = 'yolo_trash_detection/data'
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to image info mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Get list of image files
    image_files = list(annotations_by_image.keys())
    
    # Split into train and validation sets (80% train, 20% validation)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    
    # Process training set
    print("Processing training set...")
    for img_id in tqdm(train_files, desc="Training set"):
        process_image(img_id, image_id_to_info, annotations_by_image, images_dir, 
                     os.path.join(output_dir, 'images', 'train'), 
                     os.path.join(output_dir, 'labels', 'train'))
    
    # Process validation set
    print("\nProcessing validation set...")
    for img_id in tqdm(val_files, desc="Validation set"):
        process_image(img_id, image_id_to_info, annotations_by_image, images_dir, 
                     os.path.join(output_dir, 'images', 'val'), 
                     os.path.join(output_dir, 'labels', 'val'))
    
    # Create dataset.yaml file
    create_dataset_yaml(output_dir, len(coco_data['categories']))
    
    print("\nConversion complete!")

def process_image(img_id, image_id_to_info, annotations_by_image, src_images_dir, dst_images_dir, dst_labels_dir):
    """Process a single image and its annotations"""
    img_info = image_id_to_info[img_id]
    img_filename = img_info['file_name']
    img_width = img_info['width']
    img_height = img_info['height']
    
    # Copy image to destination directory
    src_img_path = os.path.join(src_images_dir, img_filename)
    dst_img_path = os.path.join(dst_images_dir, img_filename)
    
    # Create label file path (same name as image but with .txt extension)
    label_filename = os.path.splitext(img_filename)[0] + '.txt'
    label_filepath = os.path.join(dst_labels_dir, label_filename)
    
    # Skip if both image and label already exist
    if os.path.exists(dst_img_path) and os.path.exists(label_filepath):
        return
    
    # Copy image
    shutil.copy2(src_img_path, dst_img_path)
    
    # Process annotations
    annotations = annotations_by_image.get(img_id, [])
    
    # Write annotations to label file
    with open(label_filepath, 'w') as f:
        for ann in annotations:
            # COCO category_id starts from 0, YOLO uses 0-based indexing
            category_id = ann['category_id']
            
            # Convert bbox to YOLO format
            bbox = ann['bbox']
            yolo_bbox = convert_bbox_coco2yolo(img_width, img_height, bbox)
            
            # Write to file: class_id x_center y_center width height
            f.write(f"{category_id} {' '.join(map(str, yolo_bbox))}\n")

def create_dataset_yaml(output_dir, num_classes):
    """Create dataset.yaml file for YOLO training"""
    yaml_content = f"""# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {os.path.abspath(output_dir)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes (0: rubbish)
names:
  0: rubbish

# Download script/URL (optional)
download: 
"""
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset configuration at: {yaml_path}")

if __name__ == "__main__":
    main()
