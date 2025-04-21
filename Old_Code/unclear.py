import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

# Initialize the MTCNN model (you mentioned you'll add your own logic if needed)
mtcnn = MTCNN(keep_all=True, device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu')


def find_euclidean_distance(src, dst):
    """provides Euclidean distance for Face Alignment process."""
    return np.linalg.norm(src - dst)


# Placeholder for your alignment procedure. Replace with your actual alignment logic.
def alignment_procedure(face_img, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    # Find the direction to rotate the image based on the eye coordinates
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # Clockwise
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # Counter-clockwise
    # Calculate the length of the triangle edges
    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(left_eye), np.array(right_eye))
    # Apply cosine rule to find the angle
    if b != 0 and c != 0:  # Avoid division by zero
        cos_a = (b**2 + c**2 - a**2) / (2 * b * c)
        angle = np.arccos(cos_a)  # Angle in radians
        angle = np.degrees(angle)  # Convert to degrees
        # Adjust the angle based on the rotation direction
        if direction == -1:
            angle = 90 - angle
        # Rotate the image using PIL
        #img = Image.fromarray(img)
        face_img = face_img.rotate(direction * angle, resample=Image.BICUBIC)
        #img = np.array(img)  # Convert back to numpy array
        # aligned_img_pil = Image.fromarray(img)  # Convert back to PIL Image for saving
        # aligned_img_pil.show()
    return face_img

def is_blurred(face_image, threshold=100):
    # Convert PIL image to grayscale NumPy array
    gray = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def process_images(input_folder, output_folder, margin=0, blur_threshold=150):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            #print(f"Processing {image_path} ...")
            
            try:
                # Open the image and perform face detection with landmarks
                image = Image.open(image_path)
                boxes, confidences, landmarks = mtcnn.detect(image, landmarks=True)
                
                # If no faces detected, skip this image
                if boxes is None or len(boxes) == 0:
                    #print(f"No faces detected in {filename}")
                    continue
                
                # Process each detected face that passes a confidence threshold
                threshold_conf = 0.70
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    if conf < threshold_conf:
                        continue
                    
                    # Convert bounding box values to integer and add margin
                    box = [int(b) for b in box]
                    x1 = max(0, box[0] - margin)
                    y1 = max(0, box[1] - margin)
                    x2 = min(image.width, box[2] + margin)
                    y2 = min(image.height, box[3] + margin)
                    
                    # Crop the face from the image
                    cropped_face = image.crop((x1, y1, x2, y2))
                    
                    # Use landmarks to align the face (if available)
                    if landmarks is not None:
                        left_eye, right_eye = landmarks[i][0], landmarks[i][1]
                        aligned_face = alignment_procedure(cropped_face, left_eye, right_eye)
                    else:
                        aligned_face = cropped_face
                    
                    # Check if the aligned face is blurred
                    blurred = is_blurred(aligned_face, threshold=blur_threshold)
                    
                    # Prepare the output filename indicating the blur flag
                    blur_flag = "blurred" if blurred else "not_blurred"
                    base_filename, ext = os.path.splitext(filename)
                    output_filename = f"{base_filename}_face{i}_{blur_flag}{ext}"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Save the aligned face image
                    aligned_face.save(output_path)
                    #print(f"Saved face {i} from {filename} as {output_filename} (Blurred: {blurred})")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Specify your input and output folder paths
    input_folder = r'E:\2024\hires\324\Game\26515'
    output_folder = r'E:\blur'
    
    process_images(input_folder, output_folder)
