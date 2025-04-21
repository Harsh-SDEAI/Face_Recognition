import cv2
import pytesseract #type:ignore
import numpy as np

#This is the main function which receives the image and returns the digit detected in the image
def extract_jersey_number(image_path):
    """
    Extracts the jersey number from the back of a player's jersey in an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Extracted jersey number or a message if no number is detected.
    """
    # Load the image
    image = cv2.imread(image_path)
    orig = image.copy()

    # Resize the image for better processing
    (H, W) = image.shape[:2]
    newW, newH = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))

    # Load the pre-trained EAST text detector model
    net = cv2.dnn.readNet(r"D:\Python\Old_code\frozen_east_text_detection.pb")

    # Define the output layers for the EAST detector
    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ]

    # Prepare the image for the network
    blob = cv2.dnn.blobFromImage(
        image, 1.7, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)
    # Decode the predictions from the EAST detector
    (rects, confidences) = decode_predictions(scores, geometry)

    # Apply non-maxima suppression to filter overlapping boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Initialize a list to store the detected text regions
    results = []

    for (startX, startY, endX, endY) in boxes:
        # Scale the bounding box coordinates back to the original image size
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Extract the region of interest (ROI)
        roi = orig[startY:endY, startX:endX]

        # Convert ROI to grayscale and apply OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 8 digits")
        results.append(text.strip())

    # Filter results for numeric values (likely jersey numbers)
    jersey_numbers = [result for result in results if result.isdigit()]

    return jersey_numbers[0] if jersey_numbers else "No jersey number detected."

def decode_predictions(scores, geometry):
    """
    Decode the predictions from the EAST text detector.
    
    Args:
        scores (numpy.ndarray): Scores from the EAST detector.
        geometry (numpy.ndarray): Geometry data from the EAST detector.

    Returns:
        tuple: A tuple of bounding boxes and confidence scores.
    """
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def non_max_suppression(boxes, probs=None, overlapThresh=1.5):
    """
    Apply non-maxima suppression to avoid overlapping boxes.
    
    Args:
        boxes (numpy.ndarray): Array of bounding boxes.
        probs (numpy.ndarray): Array of confidence scores.
        overlapThresh (float): Threshold for overlapping boxes.

    Returns:
        list: Filtered bounding boxes.
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2 if probs is None else probs
    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Example usage
image_path = r"D:\jersey_testing\2024-JUNE 13-MONDAY-TIME 1-FIELD 1-324-GAME-26475-TOURNAMENT-88.JPG"
jersey_number = extract_jersey_number(image_path)
print(f"Detected Jersey Number: {jersey_number}")
