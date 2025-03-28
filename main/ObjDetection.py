# Requirement
# pip install opencv-python
# pip install ultralytics /  for low memmory pip install --no-cache-dir ultralytics
import numpy as np
from ultralytics import YOLO
import json

def ObjDetection(width, height, pixels):
    """
    Detect objects in an image using YOLOv8.
    
    Args:
        sub_json: JSON object containing image data in RGB format
    
    Returns:
        JSON object containing detected objects and class labels.
    """
    try:
        # Load YOLO model
        model = YOLO('yolov8n.pt')  # Load the model
    except Exception as e:
        return json.dumps({"error": f"Failed to load YOLO model: {str(e)}", "status": "ERROR"}, indent=4)
    
    try:
        # Extract image data from JSON
        # width = sub_json['image']['width']
        # height = sub_json['image']['height']
        # pixels = sub_json['image']['pixels']
        
        # Create an image array using NumPy
        image_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel = pixels[i * width + j]
                image_rgb[i, j] = [pixel['r'], pixel['g'], pixel['b']]
    except KeyError as e:
        return json.dumps({"error": f"Missing key in JSON input: {str(e)}", "status": "ERROR"}, indent=4)
    except Exception as e:
        return json.dumps({"error": f"Failed to process image data: {str(e)}", "status": "ERROR"}, indent=4)
    
    try:
        # Perform detection
        results = model(image_rgb)[0]
        
        # Generate random colors for classes
        np.random.seed(42)  # For consistent colors
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        
        # Process detections
        detected_objects = []
        for box in results.boxes:
            detected_objects.append({
                "class": results.names[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "bbox": {
                    "x": int(box.xyxy[0,0].item()),
                    "y": int(box.xyxy[0,1].item()),
                    "width": int(box.xyxy[0,2].item() - box.xyxy[0,0].item()),
                    "height": int(box.xyxy[0,3].item() - box.xyxy[0,1].item())
                },
                "color": colors[int(box.cls.item()) % len(colors)].tolist()
            })
        
        # Create the JSON object
        output_json = {
            "detected_objects": detected_objects,
            "status": "SUCCESS"
        }
        
        return json.dumps(output_json, indent=4)
    except Exception as e:
        return json.dumps({"error": f"Failed to perform object detection: {str(e)}", "status": "ERROR"}, indent=4)
