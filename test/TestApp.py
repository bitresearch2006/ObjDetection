
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
import base64

# Add the path of folder1 to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../main')))
from ObjDetection import ObjDetection

def TestApp(image_path):
    """
    Display the results of object detection.
    
    Args:
        image_path: Image path.
    """
    try:
       
        # Read original image
        original_image = cv2.imread(image_path)
        
        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Encode the image to base64
        _, buffer = cv2.imencode('.jpg', rgb_image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        # Create the sub_json object with image data
        image_json = {"image_b64": image_b64}
        output_json = ObjDetection(image_b64)

        # Parse the JSON object
        results = json.loads(output_json)
        
        # Check status
        if results.get('status') != 'SUCCESS':
            print(f"Error: {results.get('error')}")
            return
        
        detected_objects = results['detected_objects']
        
        # Draw bounding boxes and labels
        for obj in detected_objects:
            x, y, w, h = obj['bbox']['x'], obj['bbox']['y'], obj['bbox']['width'], obj['bbox']['height']
            class_name = obj['class']
            confidence = obj['confidence']
            color = obj['color']
            
            # Draw rectangle
            cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the image
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Failed to display results: {str(e)}")

# Example usage:
TestApp('test4.jpg')