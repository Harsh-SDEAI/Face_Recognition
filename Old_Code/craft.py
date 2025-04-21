import cv2
import numpy as np
from craft_text_detector import Craft #type: ignore

# Initialize the CRAFT model
craft = Craft(output_dir='./output', crop_type="poly", cuda=False)

# Preprocessing Function
def preprocess_image(image_path, target_size=(1280, 1280)):
    image = cv2.imread(image_path)
    original_image = image.copy()
    
    # Resize while maintaining aspect ratio
    h, w, _ = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    padded_image[:new_h, :new_w] = image

    return original_image, padded_image

# Draw bounding boxes
def draw_boxes(image, boxes):
    for box in boxes:
        pts = np.array(box['points'], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    return image

# Parameters
image_path = 'your_image.jpg'
preprocessed_size = (1280, 1280)

# Preprocessing
original_image, preprocessed_image = preprocess_image(image_path, preprocessed_size)

# Text Detection
prediction = craft.detect_text(preprocessed_image)

# Get bounding boxes
boxes = prediction["boxes"]

# Visualize Results
image_with_boxes = draw_boxes(original_image.copy(), boxes)

# Save and Display Results
cv2.imshow("Original Image", original_image)
cv2.imshow("Detected Text Regions", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Clean up
craft.unload_craftnet_model()
craft.unload_refinenet_model()
