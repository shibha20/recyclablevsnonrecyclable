import os
import inspect
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

# Initialize Flask app with static folder configuration
app = Flask(__name__, static_url_path='')

# Configure static folder
app.static_folder = 'static'

# Configure upload and result folders with full paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

# Ensure the upload and result folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Print absolute paths for debugging
print(f"Working directory: {os.getcwd()}")
print(f"Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
print(f"Result folder: {os.path.abspath(app.config['RESULT_FOLDER'])}")
print(f"Static folder: {os.path.abspath(app.static_folder)}")

# Initialize model as None
model = None

try:
    print("Initializing YOLO model...")
    import torch
    import sys
    from ultralytics import YOLO
    
    # Set environment variables
    import os
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # Workaround for PyTorch 2.6+ security settings
    if sys.version_info >= (3, 11) and hasattr(torch.serialization, 'safe_globals'):
        print("Using PyTorch 2.6+ safe loading workaround...")
        import functools
        import pickle
        import warnings
        
        # Monkey patch torch.load to use weights_only=False for YOLO models
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' in inspect.signature(original_torch_load).parameters:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        
        # Suppress the warning about arbitrary code execution
        warnings.filterwarnings('ignore', message='.*arbitrary code execution.*')
    
    # Load the fine-tuned model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'yolo_trash_detection', 'trash_detection', 'weights', 'finetuned.pt')
    print(f"Loading fine-tuned model from: {model_path}")
    model = YOLO(model_path, task='detect')
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Warning: Running without model - detection features will not work")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    """Process the image using YOLO model and return the annotated image path and detection info"""
    # Check if model is loaded
    if model is None:
        raise ValueError("Model failed to load. Please check the server logs.")
        
    # Generate a unique filename for the result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"result_{timestamp}.jpg"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    
    try:
        # Read the original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Failed to read image at {image_path}")
            
        # Run inference
        results = model.predict(original_img, conf=0.25)  # Use the loaded image
        
        # Process results
        detection_info = []
        
        # Create a copy of the original image for drawing
        annotated_img = original_img.copy()
        
        for result in results:
            # Draw boxes on the image
            boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
            classes = result.boxes.cls.cpu().numpy()  # Get class IDs
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
            
            for i, (box, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, box)
                
                # Get class name
                class_name = result.names[int(class_id)]
                
                # Draw rectangle
                color = (0, 255, 0)  # Green color for boxes
                thickness = 2
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
                
                # Add label with confidence
                label = f"{class_name} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add to detection info
                detection_info.append({
                    'class': class_name,
                    'confidence': f"{confidence:.2f}",
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Convert BGR to RGB for saving with PIL
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Save the result
        im = Image.fromarray(annotated_img_rgb)
        im.save(result_path)
        
        print("\nDetection Results:")
        print(f"- Image size: {original_img.shape}")
        print(f"- Detected objects: {len(detection_info)}")
        
        # Verify the file was saved
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Failed to save result image to {result_path}")
            
        # Determine detection status message
        if detection_info:
            status = "Trash detected in the image!"
            status_class = "text-success"
        else:
            status = "No trash was detected in the image."
            status_class = "text-warning"
            
        return {
            'result_filename': result_filename,
            'detections': detection_info,
            'status': status,
            'status_class': status_class
        }
            
    except Exception as e:
        print(f"Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'status': 'Error processing image',
            'status_class': 'text-danger'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Saved uploaded file to: {os.path.abspath(filepath)}")
        
        # Process the image
        result = process_image(filepath)
        
        if 'error' in result:
            return jsonify({
                'error': result['error'],
                'status': result.get('status', 'Error'),
                'status_class': result.get('status_class', 'text-danger')
            }), 500
            
        print(f"Generated result file: {result['result_filename']}")
        print(f"Detection status: {result['status']}")
        
        # Return JSON response with the result
        return jsonify({
            'result': result['result_filename'],
            'detections': result['detections'],
            'status': result['status'],
            'status_class': result['status_class']
        })
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_absolute_path(folder, filename):
    """Get absolute path to a file in the specified folder"""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_dir, folder, filename)

@app.route('/static/results/<path:filename>')
def result_file(filename):
    """Serve result images with proper caching headers"""
    try:
        # Get the absolute path to the file
        file_path = os.path.join(RESULT_FOLDER, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return "File not found", 404
            
        # Send the file with proper headers
        response = send_file(
            file_path,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
        # Prevent caching of the result image
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except Exception as e:
        print(f"Error serving file {filename}: {str(e)}")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5059)
