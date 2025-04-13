# Requirement
# pip install opencv-python
# pip install ultralytics /  for low memmory pip install --no-cache-dir ultralytics
import numpy as np
from ultralytics import YOLO
import cv2
import base64
import json

def ObjDetection(image_b64):
    """
    Detect objects in an image using YOLOv8.
    
    Args:
        image_b64: Base64 encoded image data (JPG RGB)
    
    Returns:
        JSON object containing detected objects and class labels.
    """
    try:
        # Load YOLO model
        model = YOLO('yolov8n.pt')  # Load the model
    except Exception as e:
        return json.dumps({"error": f"Failed to load YOLO model: {str(e)}", "status": "ERROR"}, indent=4)
    
    try:
        # Decode the base64 string
        image_data = base64.b64decode(image_b64)
        
        # Convert the decoded data to a NumPy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode the NumPy array to an image
        image_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to decode image data: {str(e)}", "status": "ERROR"}, indent=4)
    
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

# Example usage
# image_path = 'test4.jpg'
# with open(image_path, "rb") as image_file:
    # image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

# print(ObjDetection(image_b64))