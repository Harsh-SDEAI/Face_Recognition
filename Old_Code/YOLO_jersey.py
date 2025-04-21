from ultralytics import YOLO # type: ignore
import cv2
import pytesseract # type: ignore
from PIL import Image

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Replace with your trained model for player detection

def detect_players(image):
    results = model(image)
    # Extract bounding boxes with confidence > 0.5
    player_boxes = []
    for result in results:
        # Extract bounding boxes, confidences, and class IDs
        for box in result.boxes.data:  # Access `boxes` attribute
            x1, y1, x2, y2, conf, cls = box.tolist()
            # Filter by confidence and ensure it's a person class (e.g., class 0)
            if conf > 0.6 and int(cls) == 0:  # Assuming '0' is the class ID for 'person'
                player_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return player_boxes


def extract_jersey_number(image, boxes):
    global jersey_numbers
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    jersey_numbers = []

    for (x1, y1, x2, y2) in boxes:
        # Crop the detected player region
        roi = image[y1:y2, x1:x2]

        # Preprocess the region (resize, convert to grayscale)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Perform OCR
        text = pytesseract.image_to_string(gray, config="--psm 6 digits").strip()
        jersey_numbers.append(text)

    return jersey_numbers


def process_image(image_path):
    image = cv2.imread(image_path)

    # Step 1: Detect players
    player_boxes = detect_players(image)

    # Step 2: Extract jersey numbers
    jersey_numbers = extract_jersey_number(image, player_boxes)

    # Display results
    for idx, (box, number) in enumerate(zip(player_boxes, jersey_numbers)):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2)
        print(f"Player {idx + 1}: Jersey Number - {number}")
        image = Image.fromarray(image)
        image.show()

    # Save or display the output
    cv2.imshow("Detected Players with Jersey Numbers", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example Usage
process_image(r"D:\jersey_testing\2024-JUNE 13-MONDAY-TIME 1-FIELD 1-324-GAME-26475-TOURNAMENT-88.JPG")
