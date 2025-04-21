import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from facenet_pytorch.models.mtcnn import PNet, RNet, ONet  # Import P-Net, R-Net, O-Net
from facenet_pytorch import MTCNN
import os
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import normalize
import pyodbc as odbc
import json
from datetime import datetime
import time
import re
import mysql.connector

studio_images_folder = 'E:'  # studio photos
db_prefix_path = '\\\\172.16.17.136\\PHOTO_ROOT'

# Define fine-tuned P-Net, R-Net, and O-Net for finetuning
class FinetunedMTCNN(MTCNN):
#(self, image_size=160, margin=5, **kwargs): # use this after some time to improve the final results
    def __init__(self, **kwargs):
        super(FinetunedMTCNN, self).__init__(**kwargs)
        # Create your custom, finetuned P-Net, R-Net, O-Net here
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
    def forward(self, x):
        # Overriding forward pass if additional finetuning is needed
        return super().forward(x)
    

facenet = InceptionResnetV1(pretrained='vggface2').eval()  # Load the FaceNet model
# Initialize mtcnn model
finetuned_mtcnn = FinetunedMTCNN(keep_all=True, device='cuda:0' if torch.cuda.is_available() else 'cpu', min_face_size=20)

def find_euclidean_distance(src, dst):
    return np.linalg.norm(src - dst)

def alignment_procedure(img, left_eye, right_eye):    
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
        img = img.rotate(direction * angle, resample=Image.BICUBIC)
        img = np.array(img)  # Convert back to numpy array
        # aligned_img_pil = Image.fromarray(img)  # Convert back to PIL Image for saving
        # aligned_img_pil.show()
    return img

# Function to perform face detection and store image paths with their cropped face regions
def detect_align_embed_faces(path, mtcnn_model, margin=0):
    cropped_photos = []
    image = Image.open(path)
    boxes, confidences, landmarks = mtcnn_model.detect(image, landmarks=True)
    if boxes is not None:
        # Set a confidence threshold
        threshold = 0.95
        # Filter detected faces based on the confidence score
        filtered_faces = [i for i, confidence in enumerate(confidences) if confidence > threshold]
        # Process each filtered face
        for i in filtered_faces:
            box = boxes[i]  # Get the bounding box for the filtered face
            box = [int(b) for b in box]  # Ensure the box is in integer format
            # adding margin around the box
            # Apply margin to the bounding box
            x1 = max(0, box[0] - margin)  # Left
            y1 = max(0, box[1] - margin)  # Top
            x2 = min(image.width, box[2] + margin)  # Right
            y2 = min(image.height, box[3] + margin)  # Bottom
            # Crop the face from the image
            cropped_face = image.crop((x1, y1, x2, y2))
            if cropped_face is not None: #and cropped_face.size[0] > 170 and cropped_face.size[1] > 170:
                # Get the landmarks (left and right eyes) for the current face
                left_eye = landmarks[i][0]  # Left eye coordinates for face i
                right_eye = landmarks[i][1]  # Right eye coordinates for face i
                # Align the cropped face using the eye coordinates
                aligned_face = alignment_procedure(cropped_face, left_eye, right_eye)
                # Store the aligned face and the original image path
                cropped_photos.append(aligned_face)
        global studio_embeddings
        studio_embeddings = []
        for studio_photo in cropped_photos:
            studio_image = np.array(studio_photo) # Convert the PIL image to a NumPy array
            image = cv2.resize(studio_image, (160, 160))  # Resize to 160x160 as required by FaceNet
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Convert to tensor, normalize
            embedding = facenet(image)  # Get the embedding
            embedding = normalize(embedding, p=2, dim=1)  # L2 normalization of embeddings
            embedding = embedding.detach().numpy()
            if embedding is not None:
                studio_embeddings.append(embedding)  # Store the index and embedding
    return studio_embeddings

# # Function to get face embeddings from a pre-cropped face image
# def get_face_embedding(image):
#     image = cv2.resize(image, (160, 160))  # Resize to 160x160 as required by FaceNet
#     # aligned_img_pil = Image.fromarray(image)  # Convert back to PIL Image for saving
#     # aligned_img_pil.show()
#     image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Convert to tensor, normalize
#     embedding = facenet(image)  # Get the embedding
#     #embedding = normalize(embedding, p=2, dim=1)  # L2 normalization of embeddings
#     return embedding.detach().numpy()  # Convert to numpy array


# def store_embeddings():
#     studio_embeddings = []
#     for studio_photo in cropped_photos:
#         studio_image = np.array(studio_photo)  # Convert the PIL image to a NumPy array
#         embedding = get_face_embedding(studio_image)  # Get embedding
#         if embedding is not None:
#             studio_embeddings.append(embedding)  # Store the index and embedding
#     return studio_embeddings




# Connection to database code:
SERVER_NAME = '192.168.1.101'
DRIVER_NAME = 'ODBC Driver 17 for SQL Server'
DATABASE_NAME = 'AI_Face_Recognition'
cnxn = odbc.connect('DRIVER={ODBC Driver 17 for SQL Server}; \
                    SERVER='+SERVER_NAME+'; \
                    DATABASE = '+DATABASE_NAME+'; \
                    Uid=sa;Pwd=Masterly@123;')
cur = cnxn.cursor()

cur.execute("use AI_Face_Recognition")
#cur.execute("use CDPMediaCapture")



queue_query = "SELECT * FROM AITournamentQueue"

# Run the loop while the elapsed time is less than the specified duration.
x = True
while x is True:
    # Perform your task here
    cur.execute(queue_query)
    columns = [column[0] for column in cur.description]
    queue_rows = cur.fetchall()
    # Store the fetched rows in a Python variable (e.g., a list of dictionaries)
    queue_data = []
    for queue_row in queue_rows:
        # Assuming row is a tuple with the same column order as in the query
        queue_row_dict = {
            'AITournamentQueueId': queue_row[0],
            'TournamentID': queue_row[1],
            'GameNumber': queue_row[2],
            'Teamkey1': queue_row[3],
            'Teamkey2': queue_row[4],
            'Status': queue_row[5],
            'ErrorMessage': queue_row[6],
            'ProcessStartOn': queue_row[7],
            'ProcessEndOn': queue_row[8],
            'UpdatedBy': queue_row[9],
            'CreatedBy': queue_row[10],
            'UpdatedOn': queue_row[11],
            'CreatedOn': queue_row[12]
        }
        queue_data.append(queue_row_dict)
    # print(len(queue_data))

    for queueItem in queue_data:
        # Define the SQL query using these values
        roster_query = """select * from CDPMediaCapture.dbo.Constellation 
    where PhotoType = 'Studio' and TournamentID = {0} and MediaType = '.jpg'
    and TeamKeys in ('{1}','{2}') and PhotoUse = ''""".format(queueItem['TournamentID'], queueItem['Teamkey1'],queueItem['Teamkey2'])
        cur.execute(roster_query)
        columns = [column[0] for column in cur.description]
        roster_rows = cur.fetchall()
        for roster_row in roster_rows:
            roster_data = dict(zip(columns, roster_row))
            imageStoredPath = roster_data['KioskHiresFile'].replace(db_prefix_path, studio_images_folder)
            detect_align_embed_faces(imageStoredPath, finetuned_mtcnn, margin=100)
            for embedding in studio_embeddings:
                # Extract RosterID, TeamKey, and TournamentID
                roster_id = roster_data["RosterID"]
                constellation_id = roster_data["ConstellationID"]
                team_key = roster_data["TeamKeys"]
                tournament_id = roster_data["TournamentID"]
                # Check if the embedding for this RosterID and TournamentID already exists
                cur.execute("""
                    SELECT COUNT(*) FROM PlayerPhotoEmbedding
                    WHERE RosterID = ? AND TournamentID = ?
                """, (roster_id, tournament_id))
                result = cur.fetchone()
                if result[0] > 0:
                    pass
                else:
                    # Insert data into PlayerPhotoEmbedding table
                    cur.execute("""
                        INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                    """, (roster_id, tournament_id, str(embedding), imageStoredPath, team_key, constellation_id))  # Convert embedding to string
                    cnxn.commit()
                    print(f"Inserted embedding for RosterID {roster_id} and TournamentID {tournament_id}.")


    print("Complete.")
    x = False







