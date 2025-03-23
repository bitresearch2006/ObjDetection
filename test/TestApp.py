
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

# Add the path of folder1 to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../main')))
from ObjectDetection import ObjectDetection

def image_to_json(image_path):
    """
    Convert an image to a JSON object with RGB pixel data.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        JSON object containing image data in RGB format
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width, _ = image_rgb.shape
        
        # Create a list to hold pixel data
        pixels = [{"r": int(r), "g": int(g), "b": int(b)} for row in image_rgb for r, g, b in row]
        
        # Create the JSON object
        image_json = {
            "image": {
                "width": width,
                "height": height,
                "pixels": pixels
            }
        }
        
        return image_json
    except Exception as e:
        print(f"Failed to convert image to JSON: {str(e)}")

def TestApp(image_path, output_json):
    """
    Display the results of object detection.
    
    Args:
        output_json: JSON object containing detected objects and class labels.
    """
    try:
        # Parse the JSON object
        results = json.loads(output_json)
        
        # Check status
        if results.get('status') != 'SUCCESS':
            print(f"Error: {results.get('error')}")
            return
        
        detected_objects = results['detected_objects']
        
        # Read original image
        original_image = cv2.imread(image_path)
        annotated_image = original_image
        
        # Draw bounding boxes and labels
        for obj in detected_objects:
            x, y, w, h = obj['bbox']['x'], obj['bbox']['y'], obj['bbox']['width'], obj['bbox']['height']
            class_name = obj['class']
            confidence = obj['confidence']
            color = obj['color']
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the image
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Failed to display results: {str(e)}")

# Example usage:
image_json = image_to_json('test4.jpg')
output_json = ObjectDetection(image_json)
print(output_json)
TestApp('test4.jpg', output_json)