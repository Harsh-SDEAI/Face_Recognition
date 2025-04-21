import cv2
import pytesseract #type:ignore
import numpy as np
import os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
count = 0
def extract_jersey_number_from_folder(folder_path):
    global count
    net = cv2.dnn.readNet(r"D:\Python\Old_code\frozen_east_text_detection.pb")

    # Define the output layers for the EAST detector
    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ]

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if not (filename.lower().endswith(('.jpg'))):
            continue

        # Load the image
        image = cv2.imread(image_path)
        orig = image.copy()

        # Resize the image for better processing
        (H, W) = image.shape[:2]
        newW, newH = (640, 640)
        rW = W / float(newW)
        rH = H / float(newH)
        image = cv2.resize(image, (newW, newH))

        # Prepare the image for the network
        blob = cv2.dnn.blobFromImage(
            image, 1.7, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=True
        )
        net.setInput(blob)
        (scores, geometry) = net.forward(layer_names)

        # Decode the predictions from the EAST detector
        (rects, confidences) = decode_predictions(scores, geometry)

        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # Initialize a list to store the detected text regions
        results = []
        margin = 0
        for (startX, startY, endX, endY) in boxes:
            # Scale the bounding box coordinates back to the original image size
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # Clip coordinates to ensure they are within the image bounds
            startX = max(0, startX - margin)
            startY = max(0, startY - margin)
            endX = min(W, endX + margin)
            endY = min(H, endY + margin)
            # Ensure valid bounding box dimensions
            if startX >= endX or startY >= endY:
                print(f"Invalid bounding box detected: ({startX}, {startY}, {endX}, {endY})")
                continue
            # Extract the region of interest (ROI)
            roi = orig[startY:endY, startX:endX]
            # Check if the ROI is empty
            if roi is None or roi.size == 0:
                print(f"Empty ROI detected for bounding box: ({startX}, {startY}, {endX}, {endY})")
                continue
            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # Apply Otsu's thresholding
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Optional: You can also apply additional cleaning such as dilation
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(binary, kernel, iterations=1)
                text = pytesseract.image_to_string(dilated, config="--psm 6 digits")
                results.append(text.strip())
            except cv2.error as e:
                print(f"Error processing ROI: {e}")
                continue

        # Filter results for numeric values (likely jersey numbers)
        jersey_numbers = [result for result in results if result.isdigit()]
        
        # Print all detected jersey numbers for the file
        if jersey_numbers:
            print(f"File: {filename}, Detected Jersey Numbers: {jersey_numbers}")
            count += 1
        else:
            print(f"File: {filename}, No jersey number detected")


def decode_predictions(scores, geometry):
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
            if scoresData[x] < 0.1:
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

def non_max_suppression(boxes, probs=None, overlapThresh=1.1):
    if len(boxes) == 0:
        return []

    # Checks if the boxes are int type, if then converts it into float for the better precision
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
folder_path = r"D:\jersey_images"
extract_jersey_number_from_folder(folder_path)
print(count)

