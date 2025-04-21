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

studio_images_folder = r'E:\2024\hires\324\Studio\20240222'  # studio photos

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
finetuned_mtcnn = FinetunedMTCNN(keep_all=True, device='cuda:0' if torch.cuda.is_available() else 'cpu', min_face_size=60)

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
def detect_and_align_faces(image_folder, mtcnn_model, margin=0):
    global studio_photos_all, studio_photos_paths
    studio_photos_all = []
    studio_photos_paths = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".JPG"):
            #print("1")
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            boxes, confidences, landmarks = mtcnn_model.detect(image, landmarks=True)
            if boxes is not None:
                # Set a confidence threshold
                threshold = 0.96
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
                        studio_photos_all.append(aligned_face)
                        studio_photos_paths.append(image_path)  # Store original image path
                        #print("2")
    return studio_photos_paths  # Return list of original image paths

# Function to get face embeddings from a pre-cropped face image
def get_face_embedding(image):
    image = cv2.resize(image, (160, 160))  # Resize to 160x160 as required by FaceNet
    # aligned_img_pil = Image.fromarray(image)  # Convert back to PIL Image for saving
    # aligned_img_pil.show()
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Convert to tensor, normalize
    embedding = facenet(image)  # Get the embedding
    #embedding = normalize(embedding, p=2, dim=1)  # L2 normalization of embeddings
    return embedding.detach().numpy()  # Convert to numpy array


def store_embeddings():
    global studio_embeddings
    studio_embeddings = []
    for studio_photo in studio_photos_all:
        studio_image = np.array(studio_photo)  # Convert the PIL image to a NumPy array
        embedding = get_face_embedding(studio_image)  # Get embedding
        if embedding is not None:
            studio_embeddings.append(embedding)  # Store the index and embedding
    return studio_embeddings


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



queue_query = """
    SELECT AITournamentQueueId, TournamentID, GameNumber, Teamkey1, Teamkey2, Status, ErrorMessage,
           ProcessStartOn, ProcessEndOn, UpdatedBy, CreatedBy, UpdatedOn, CreatedOn
    FROM AITournamentQueue
"""
# Run the loop while the elapsed time is less than the specified duration
x = True
while x is True:
    # Perform your task here
    cur.execute(queue_query)
    rows = cur.fetchall()
    # Store the fetched rows in a Python variable (e.g., a list of dictionaries)
    queue_data = []
    for row in rows:
        # Assuming row is a tuple with the same column order as in the query
        row_dict = {
            'AITournamentQueueId': row[0],
            'TournamentID': row[1],
            'GameNumber': row[2],
            'Teamkey1': row[3],
            'Teamkey2': row[4],
            'Status': row[5],
            'ErrorMessage': row[6],
            'ProcessStartOn': row[7],
            'ProcessEndOn': row[8],
            'UpdatedBy': row[9],
            'CreatedBy': row[10],
            'UpdatedOn': row[11],
            'CreatedOn': row[12]
        }
        queue_data.append(row_dict)
        Queue_length = len(queue_data)
        #print(f"{Queue_length} Record found in the Queue.")
        #time.sleep(2)
        # for row in rows:
        #     print(row)

        # for entry in data:
        #         print(entry)
        # To print the data of each column
        # for i in data:
        #     print("_" * 40)
        #     for j in i:
        #         print(f"{j}:{i[j]}")
    first_entry = queue_data[0]
    tournament_id = first_entry['TournamentID']
    teamkey1 = first_entry['Teamkey1']
    #print(teamkey1)
    teamkey2 = first_entry['Teamkey2']
    #print(teamkey2)
    # Define the SQL query using these values
    roster_query = f"""select RosterID,TeamKeys,* from CDPMediaCapture.dbo.Constellation 
where PhotoType = 'Studio' and TournamentID = {tournament_id} and MediaType = '.jpg'
and TeamKeys in ('{teamkey1}','{teamkey2}') and PhotoUse = ''"""
    roster_data = []
    cur.execute(roster_query)
    rows = cur.fetchall()
    for row in rows:
        row_dict = {
            'RosterID': row[0],
            'TeamKeys': row[1],
            'ConstellationID': row[2],
            'TournamentID': row[20],
            'Image_Path_DB': row[57],
            'GameNumber': row[30]
        }
        roster_data.append(row_dict)
    detect_and_align_faces(studio_images_folder, finetuned_mtcnn, margin=30)
    store_embeddings()
    #print(len(studio_photos_paths))
    #print(len(studio_embeddings))
    # Studio_photos_paths[]
    # Studio_embeddings[]
    # We need to store these two things along with the other information in the table playerEmbeddings
    # Iterate over the three lists
    for photo_path, embedding, roster_entry in zip(studio_photos_paths, studio_embeddings, roster_data):
        # Extract RosterID, TeamKey, and TournamentID
        roster_id = roster_entry["RosterID"]
        constellation_id = roster_entry["ConstellationID"]
        team_key = roster_entry["TeamKeys"]
        tournament_id = roster_entry["TournamentID"]
        # Insert data into PlayerEmbeddings table
        cur.execute("""
        INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
        """, (roster_id, tournament_id, str(embedding), photo_path, team_key, constellation_id))  # Convert embedding to string
        cnxn.commit()

        

    print("Complete.")
    x = False




    #print(roster_data)
    # count = 0
    # for i in roster_data:
    #     print("_ " * 40)
    #     for j in i:
    #         print(f"{j}:{i[j]}")
    #     count += 1
    #     print("row_number:", count)
    # # Fetch and process the query result
    # print("_ " * 40)
    # print("number of rows fetched: ", len(rows))
    # Define fine-tuned P-Net, R-Net, and O-Net for finetuning


    


