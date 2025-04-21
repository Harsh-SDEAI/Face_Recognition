# Below is the list of libraries
import torch
import numpy as np
from PIL import Image
from facenet_pytorch.models.mtcnn import PNet, RNet, ONet
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
import cv2
from torch.nn.functional import normalize
import pyodbc as odbc
import os
import time
from tqdm import tqdm



# Database Credentials
SERVER_NAME = '192.168.1.101' 
DRIVER_NAME = 'ODBC Driver 17 for SQL Server'
DATABASE_NAME = 'AI_Face_Recognition'
USER_NAME = 'sa' 
PASSWORD = 'Masterly@123'

# Parameters 
local_image_path = 'E:'                             # Where your photos are stored locally
db_prefix_path = '\\\\172.16.17.136\\PHOTO_ROOT'    # Will be replaced with the database path
min_face_size = 30                                  # Minimum faces you are allowing for the process eg. 30 (30*30 pixels)
idle_sleeptime = 1200                               # How many seconds do you want the while to sleep when there is no row found in the Queue data
processing_sleeptime = 5                            # How many seconds do you want the while to sleep when one game's process ends and another game's starts
retry_count = 3                                     # How many number of tries are you allowing to each game when they face any error
timestamp = 1200                                    # If any game has status InProgress, after how many seconds do you want to give that game a try
margin = 44                                         # How much more of the face do you want in the studio images eg. 80 (from each side it will take 80 more pixels)
euclidean_threshold = 0.87                          # Matching threshold (Distances less than threshold will get picked as matched)


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
    

def find_euclidean_distance(src, dst):
    """provides Euclidean distance for Face Alignment process."""
    return np.linalg.norm(src - dst)

def alignment_procedure(img, left_eye, right_eye): 
    """function takes the cropped face and returns aligned photo."""   
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


def detect_align_embed_studio_faces(path, mtcnn_model, margin=0):
    """function takes a photo and provides embedding for the same."""
    global error_occurred, studio_coordinates
    cropped_photos = []
    # Check if the file exists at the given path
    if not os.path.exists(path):
        error_occurred = True
        return cropped_photos 
    try:
        image = Image.open(path)
        boxes, confidences, landmarks = mtcnn_model.detect(image, landmarks=True)
        global photo_embeddings
        photo_embeddings = []
        if boxes is None:
            cur.execute("""
            SELECT COUNT(*) FROM PlayerPhotoEmbedding
            WHERE RosterID = ? AND TournamentID = ?
            """, (roster_data["RosterID"], roster_data["TournamentID"]))
            result = cur.fetchone()
            if result[0] > 0:
                pass
            else:
                cur.execute("""
                    INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID)
                    VALUES (?, ?, NULL, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """, (roster_data["RosterID"], roster_data["TournamentID"], studio_image_path, roster_data["TeamKeys"], roster_data["ConstellationID"]))
                cnxn.commit()
        else:
            threshold = 0.70
            # Filter detected faces based on the confidence score
            filtered_faces = [i for i, confidence in enumerate(confidences) if confidence > threshold]
            if filtered_faces is None:
                cur.execute("""
                SELECT COUNT(*) FROM PlayerPhotoEmbedding
                WHERE RosterID = ? AND TournamentID = ?
                """, (roster_data["RosterID"], roster_data["TournamentID"]))
                result = cur.fetchone()
                if result[0] > 0:
                    pass
                else:
                    cur.execute("""
                    INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID)
                    VALUES (?, ?, NULL, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """, (roster_data["RosterID"], roster_data["TournamentID"], studio_image_path, roster_data["TeamKeys"], roster_data["ConstellationID"]))
                    cnxn.commit()
            else:
                # Process each filtered face
                for i in filtered_faces:
                    box = boxes[i]  # Get the bounding box for the filtered face
                    box = [int(b) for b in box]  # Ensure the box is in integer format
                    # Apply margin to the bounding box
                    x1 = max(0, box[0] - margin)  # Left
                    y1 = max(0, box[1] - margin)  # Top
                    x2 = min(image.width, box[2] + margin)  # Right
                    y2 = min(image.height, box[3] + margin)  # Bottom
                    studio_coordinates = f"{x1}, {y1}, {x2}, {y2}"
                    # Crop the face from the image
                    cropped_face = image.crop((x1, y1, x2, y2))
                    if cropped_face is not None: 
                        # Get the landmarks (left and right eyes) for the current face
                        left_eye = landmarks[i][0]  
                        right_eye = landmarks[i][1] 
                        # Align the cropped face using the eye coordinates
                        aligned_face = alignment_procedure(cropped_face, left_eye, right_eye)
                        cropped_photos.append(aligned_face)
                for photo in cropped_photos:
                    face_image = np.array(photo) # Convert the PIL image to a NumPy array
                    image = cv2.resize(face_image, (160, 160))  # Resize to 160x160 as required by FaceNet
                    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Convert to tensor, normalize
                    #show_image = Image.fromarray(face_image)
                    #show_image.show()
                    embedding = facenet(image)  # Get the embedding
                    embedding = normalize(embedding, p=2, dim=1)  # L2 normalization of embeddings
                    embedding = embedding.detach().numpy()
                    if embedding is not None:
                        photo_embeddings.append(embedding)  # Store the index and embedding
            return photo_embeddings
    except Exception as e:
        # Update game status to error
        print(e)
        cur.execute("""UPDATE AITournamentQueue SET Status = 'error', RetryCount = RetryCount + 1 WHERE GameNumber = ?""", (queue_data["GameNumber"],))
        cnxn.commit()
    return cropped_photos





def detect_align_embed_game_faces(path, mtcnn_model, margin=0):
    """function takes a photo and provides embedding for the same."""
    global error_occurred, isgamephotonull, game_coordinates
    cropped_photos = []
    # Check if the file exists at the given path
    if not os.path.exists(path):
        error_occurred = True
        return cropped_photos 
    try:
        image = Image.open(path)
        boxes, confidences, landmarks = mtcnn_model.detect(image, landmarks=True)
        global photo_embeddings
        photo_embeddings = []
        if boxes is None:
            isgamephotonull = True
        else:
            # Set a confidence threshold
            threshold = 0.96
            # Filter detected faces based on the confidence score
            filtered_faces = [i for i, confidence in enumerate(confidences) if confidence > threshold]
            if filtered_faces is None:
                isgamephotonull = True
            else:
                for i in filtered_faces:
                    box = boxes[i]  # Get the bounding box for the filtered face
                    box = [int(b) for b in box]  # Ensure the box is in integer format
                    # Apply margin to the bounding box
                    x1 = max(0, box[0] - margin)  # Left
                    y1 = max(0, box[1] - margin)  # Top
                    x2 = min(image.width, box[2] + margin)  # Right
                    y2 = min(image.height, box[3] + margin)  # Bottom
                    game_coordinates = f"({x1}, {y1}), ({x2}, {y2})"
                    # Crop the face from the image
                    cropped_face = image.crop((x1, y1, x2, y2))
                    if cropped_face is not None: 
                        # Get the landmarks (left and right eyes) for the current face
                        left_eye = landmarks[i][0]  
                        right_eye = landmarks[i][1] 
                        # Align the cropped face using the eye coordinates
                        aligned_face = alignment_procedure(cropped_face, left_eye, right_eye)
                        cropped_photos.append(aligned_face)
                for photo in cropped_photos:
                    face_image = np.array(photo) # Convert the PIL image to a NumPy array
                    image = cv2.resize(face_image, (160, 160))  # Resize to 160x160 as required by FaceNet
                    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Convert to tensor, normalize
                    embedding = facenet(image)  # Get the embedding
                    embedding = normalize(embedding, p=2, dim=1)  # L2 normalization of embeddings
                    embedding = embedding.detach().numpy()
                    if embedding is not None:
                        photo_embeddings.append(embedding)  # Store the index and embedding
            return photo_embeddings
    except Exception as e:
        # Update game status to error
        print(e)
        cur.execute("""UPDATE AITournamentQueue SET Status = 'error', RetryCount = RetryCount + 1 WHERE GameNumber = ?""", (queue_data["GameNumber"],))
        cnxn.commit()
    return cropped_photos  

def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)



# Initializing the Facenet model
facenet = InceptionResnetV1(pretrained='vggface2').eval() 

# Initializing the MTCNN model
finetuned_mtcnn = FinetunedMTCNN(keep_all=True, device='cuda:0' if torch.cuda.is_available() else 'cpu', min_face_size=min_face_size)

# Connection to Database:
cnxn = odbc.connect('DRIVER='+DRIVER_NAME+'; \
                    SERVER='+SERVER_NAME+'; \
                    DATABASE = '+DATABASE_NAME+'; \
                    Uid='+USER_NAME+';Pwd='+PASSWORD+';')
cur = cnxn.cursor()
cur.execute("use AI_Face_Recognition") 

# Main service starts from here
isgamephotonull = False
error_occurred = False
while True:
    queue_count_query = "select * from AITournamentQueue where Status in ('pending', 'error', 'InProgress')"
    cur.execute(queue_count_query)
    queue_count = len(cur.fetchall())
    print("Number of Games in Queue: ", queue_count)
    queue_query = """SELECT TOP 1 * FROM AITournamentQueue WHERE (Status IN('pending', 'error') AND RetryCount < ?) 
                        OR (Status = 'InProgress' AND DATEDIFF(SECOND, ProcessStartOn, CURRENT_TIMESTAMP) > ?)"""
    cur.execute(queue_query, (retry_count, timestamp))
    queue_row_count = len(cur.fetchall())
    if queue_row_count == 0:
        time.sleep(idle_sleeptime)
    else:
        try:
            cur.execute(queue_query, (retry_count, timestamp))
            queue_columns = [column[0] for column in cur.description]
            queue_rows = cur.fetchall()
            # This will fetch the games which have status pending
            for queue_row in queue_rows:
                queue_data = dict(zip(queue_columns, queue_row))
                cur.execute("""UPDATE AITournamentQueue 
                            SET ProcessStartOn = CURRENT_TIMESTAMP, 
                                Status = 'InProgress' WHERE GameNumber = ?""", (queue_data["GameNumber"],))
                cnxn.commit()
                print(f"Photo matching process begins for Game Number: {queue_data['GameNumber']}")
                print()
                # Below query will return the studio photos for the particular game
                print(f"Studio photos embedding task initiated.")
                roster_query = """
                    SELECT * FROM CDPMediaCapture.dbo.Constellation 
                    WHERE PhotoType = 'Studio' AND TournamentID = ? AND MediaType = '.jpg' AND TeamKeys IN (?, ?) AND PhotoUse = ''"""
                cur.execute(roster_query, (queue_data['TournamentID'], queue_data['TeamKey1'], queue_data['TeamKey2']))
                roster_columns = [column[0] for column in cur.description]
                roster_rows = cur.fetchall()
                total_studio_images = len(roster_rows)
                # for each studio photos, we will find embedding and adding it to the photoembedding table
                for roster_row in tqdm(roster_rows, desc="Processing Studio Images", unit="image", total=total_studio_images):
                    roster_data = dict(zip(roster_columns, roster_row))
                    studio_image_path = roster_data['KioskHiresFile'].replace(db_prefix_path, local_image_path)
                    photo_embeddings = detect_align_embed_studio_faces(studio_image_path, finetuned_mtcnn, margin=margin)
                    # Checking if the embedding is already present or not for the specific studio photo
                    for embedding in photo_embeddings:
                        cur.execute("""
                            SELECT COUNT(*) FROM PlayerPhotoEmbedding
                            WHERE RosterID = ? AND TournamentID = ? AND BoundingBox is not null
                        """, (roster_data["RosterID"], roster_data["TournamentID"]))
                        result = cur.fetchone()
                        if result[0] > 0:
                            pass
                        else:
                            cur.execute("""
                                INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID, BoundingBox)
                                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                            """, (roster_data["RosterID"], roster_data["TournamentID"], embedding.tobytes(), studio_image_path, roster_data["TeamKeys"], roster_data["ConstellationID"], studio_coordinates))
                            cnxn.commit()
                print(f"Studio photos embedding task completed.")
                print()
                # Below query checks if there are already game photo embeddings exist in the GamePhotoEmbedding table, if found, it deletes them.
                cur.execute("DELETE FROM GamePhotoDetail WHERE GameNumber = ?", (queue_data["GameNumber"],))
                cur.execute("DELETE FROM GamePhotoEmbedding WHERE GameNumber = ?", (queue_data["GameNumber"],))
                # Below query will provide the game photos respective to the game number
                print(f"Game photos embedding task initiated.")
                game_query = """SELECT * FROM CDPMediaCapture.dbo.Constellation WHERE MediaType = '.jpg' AND GameNumber = ?"""
                cur.execute(game_query, (queue_data["GameNumber"],))
                game_columns = [column[0] for column in cur.description]
                game_rows = cur.fetchall()
                total_game_images = len(game_rows)
                # For each game photo, finding the number of faces and adding the details in the game photo detail table
                game_photos = 0
                for game_row in tqdm(game_rows, desc="Processing Game Images", unit="image", total=total_game_images):
                    game_data = dict(zip(game_columns, game_row))
                    game_image_path = game_data['KioskHiresFile'].replace(db_prefix_path, local_image_path)
                    photo_embeddings = detect_align_embed_game_faces(game_image_path, finetuned_mtcnn, margin=margin)
                    id = cur.execute("""
                        INSERT INTO GamePhotoDetail (NuFaceDetected, ImagePath, RosterIDs, GameNumber, TournamentID, TournamentWeek, TournamentWeekNumber, GameDay, GameTime, GameField, GameDayNumber, GameTimeNumber, GameType, U, B, R, ConstellationID, IsMatch, IsGroupPhoto, IsTagged, IsAttempted, UpdatedOn, CreatedOn) OUTPUT inserted.GamePhotoDetailID
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (len(photo_embeddings), game_image_path, None, game_data["GameNumber"], game_data["TournamentID"], game_data["TournamentWeek"], game_data["TournamentWeekNumber"], game_data["GameDay"], game_data["GameTime"], game_data["GameField"], game_data["GameDayNumber"], game_data["GameTimeNumber"], game_data["GameType"], game_data["U"], game_data["B"], game_data["R"], game_data["ConstellationID"])).fetchval()
                    cnxn.commit()
                    if isgamephotonull is True:
                        cur.execute("""
                            INSERT INTO GamePhotoEmbedding (GamePhotoDetailID, GFaceEmbedding, UpdatedOn, CreatedOn, GameNumber)
                            VALUES (?, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                        """, (id, game_data['GameNumber']))
                        cnxn.commit()
                        isgamephotonull = False
                    # For each game photo, adding the embedding and gamephotodetailid into the gamephotoembedding table
                    for embedding in photo_embeddings:
                        game_photos += 1
                        cur.execute("""
                            INSERT INTO GamePhotoEmbedding (GamePhotoDetailID, GFaceEmbedding, UpdatedOn, CreatedOn, GameNumber, BoundingBox)
                            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                        """, (id, embedding.tobytes(), game_data['GameNumber'], game_coordinates))
                        cnxn.commit()
                print(f"Game photos embedding task completed.")
                print()
                print(f"Matching task started.")
                cur.execute("""SELECT PlayerEmbeddingId, SFaceEmbeddings, RosterID FROM PlayerPhotoEmbedding WHERE TeamKey IN (?, ?) AND SFaceEmbeddings is Not Null""", (queue_data['TeamKey1'], queue_data['TeamKey2']))
                player_columns = [column[0] for column in cur.description]
                player_embeddings = [dict(zip(player_columns, row)) for row in cur.fetchall()]
                # Process each studio embedding
                match_count = 0
                for player in player_embeddings:
                    s_embedding = np.frombuffer(player['SFaceEmbeddings'], dtype=np.float32).reshape(1, -1)
                    # Fetch game embeddings with the same GameNumber
                    cur.execute("""
                        SELECT GamePhotoEmbeddingId, GFaceEmbedding, GamePhotoDetailId 
                        FROM GamePhotoEmbedding 
                        WHERE GameNumber = ? AND GFaceEmbedding is Not Null
                    """, (game_data['GameNumber'],))
                    game_columns = [column[0] for column in cur.description]
                    game_embeddings = [dict(zip(game_columns, row)) for row in cur.fetchall()]
                    # Compare with relevant game embeddings
                    for game in game_embeddings:
                        g_embedding = np.frombuffer(game['GFaceEmbedding'], dtype=np.float32).reshape(1, -1)
                        distance = euclidean_distance(s_embedding, g_embedding)
                        if distance < euclidean_threshold:
                            match_count+=1
                            cur.execute("""
                                        UPDATE GamePhotoDetail
                                        SET MatchedFaces = MatchedFaces + 1,
                                            RosterIDs = COALESCE(RosterIDs + ' ', '') + ?
                                        WHERE GamePhotoDetailId = ?
                                    """, (str(player['RosterID']), game['GamePhotoDetailId']))
                            cnxn.commit()  
            print(f"{match_count} photos matched out of {game_photos} Game photos.")
            print("Matching task completed.")
            print()
            print(f"Photo matching process completed for Game Number: {queue_data['GameNumber']}")
            print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            if error_occurred is False:
                cur.execute("""UPDATE AITournamentQueue SET Status = 'completed', ProcessEndOn = CURRENT_TIMESTAMP WHERE GameNumber = ?""", (queue_data['GameNumber'],))
                cnxn.commit()
            else:
                cur.execute("""UPDATE AITournamentQueue SET Status = 'error', ProcessEndOn = CURRENT_TIMESTAMP, RetryCount = RetryCount + 1 WHERE GameNumber = ?""", (queue_data['GameNumber'],))
                cnxn.commit()
                error_occurred = False
            time.sleep(processing_sleeptime)

        except Exception as e:
            print(
                type(e).__name__,          # TypeError
                __file__,                  # /tmp/example.py
                e.__traceback__.tb_lineno  # 2
            )
            print(e)
            cur.execute("""UPDATE AITournamentQueue SET Status = 'error', ProcessEndOn = CURRENT_TIMESTAMP, RetryCount = RetryCount + 1 WHERE GameNumber = ?""", (queue_data['GameNumber'],))
            cnxn.commit()
            break