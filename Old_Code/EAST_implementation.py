import cv2
import numpy as np
import matplotlib.pyplot as plt

def decode_predictions(scores, geometry, conf_threshold=0.5):
    """Decodes the predictions from the EAST model into bounding boxes."""
    detections = []
    confidences = []

    height, width = scores.shape[2:4]
    for y in range(height):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(width):
            if scores_data[x] < conf_threshold:
                continue

            # Get the offset values
            offset_x = x * 4.0
            offset_y = y * 4.0

            # Angle and cosine/sine
            angle = angles_data[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Width and height of the bounding box
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            # Calculate bounding box coordinates
            end_x = int(offset_x + cos_a * x_data1[x] + sin_a * x_data2[x])
            end_y = int(offset_y - sin_a * x_data1[x] + cos_a * x_data2[x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            detections.append((start_x, start_y, end_x, end_y))
            confidences.append(float(scores_data[x]))

    return detections, confidences

def draw_boxes(image, boxes, confidences, conf_threshold=0.5):
    """Draw bounding boxes on the image."""
    for (box, confidence) in zip(boxes, confidences):
        if confidence >= conf_threshold:
            start_x, start_y, end_x, end_y = box
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)

# Parameters to tune
conf_threshold = 0.3   # Confidence threshold
nms_threshold = 0.4  # Non-maximum suppression threshold
input_height = 128    # Height of the input image to the model
input_width = 128   # Width of the input image to the model

# Load the pre-trained EAST model
model_path = r"D:\Python\Old_code\frozen_east_text_detection.pb"
net = cv2.dnn.readNet(model_path)

# Load the input image
image_path = r"C:\Users\harsh.n\Desktop\Extension.jpg" 
image = cv2.imread(image_path)
original_image = image.copy()
height, width, _ = image.shape
print("Input image dimensions:", image.shape)
# Prepare the image for the model
blob = cv2.dnn.blobFromImage(image, 1.0, (input_width, input_height), (123.68, 116.78, 103.94), swapRB=True, crop=True)
net.setInput(blob)
print("Blob shape:", blob.shape)

# Get model output
layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
scores, geometry = net.forward(layer_names)
print("Scores shape:", scores.shape)
print("Geometry shape:", geometry.shape)

# Decode the predictions
boxes, confidences = decode_predictions(scores, geometry, conf_threshold)
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
print("Indices after NMS:", indices)

# Draw detected boxes on the original image
detected_boxes = [boxes[i] for i in indices]
final_confidences = [confidences[i] for i in indices]
draw_boxes(original_image, detected_boxes, final_confidences, conf_threshold)

# Visualize the results
plt.figure(figsize=(12, 12))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Detected Boxes")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Score Map")
plt.imshow(scores[0, 0, :, :], cmap="hot")

plt.show()

# Print geometries and box details
print("Detected Boxes and Confidence Scores:")
for box, confidence in zip(detected_boxes, final_confidences):
    print(f"Box: {box}, Confidence: {confidence}")