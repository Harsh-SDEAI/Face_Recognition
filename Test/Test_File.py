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
import easyocr
from matplotlib import pyplot as plt
import logging
import cv2
import traceback

# Suppress logging messages from EasyOCR
logging.getLogger("easyocr").setLevel(logging.ERROR)

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
idle_sleeptime = 60                                 # How many seconds do you want the while to sleep when there is no row found in the Queue data
processing_sleeptime = 5                            # How many seconds do you want the while to sleep when one game's process ends and another game's starts
retry_count = 3                                     # How many number of tries are you allowing to each game when they face any error
timestamp = 30                                      # If any game has status InProgress, after how many seconds do you want to give that game a try
margin = 44                                         # How much more of the face do you want in the studio images eg. 80 (from each side it will take 80 more pixels)


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
                    studio_coordinates = f"({x1}, {y1}), ({x2}, {y2})"
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
        print(e)
    return cropped_photos  

def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# def draw_bounding_boxes(img_path, results):
#     # Load the image
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError(f"Image not found at path: {img_path}")
#     # Set font and spacer for text placement
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     spacer = 20  # Initial spacer for vertical text placement
#     # Ensure results is a list (even if only one detection is provided)
#     if not isinstance(results, list):
#         raise ValueError("Results should be a list of detections.")
#     # Loop through detections
#     for detection in results:
#         # Ensure detection has three elements: bounding box, text, and confidence
#         if len(detection) != 3:
#             raise ValueError("Each detection should have bounding box, text, and confidence.")
#         # Extract bounding box, text, and confidence
#         bbox = detection[0]
#         text = detection[1]
#         confidence = detection[2]
#         # Ensure bounding box has 4 points
#         if len(bbox) != 4:
#             raise ValueError("Bounding box should have 4 points.")
#         # Extract coordinates for rectangle
#         top_left = tuple(map(int, bbox[0]))
#         bottom_right = tuple(map(int, bbox[2]))
#         # Draw rectangle
#         img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
#         # Draw text with confidence
#         display_text = f"{text} ({confidence:.2f})"
#         img = cv2.putText(img, display_text, (top_left[0], top_left[1] - 10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
#         # Optionally display all text stacked below the image
#         img = cv2.putText(img, display_text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
#         spacer += 20
#     # Return the image with drawn bounding boxes
#     return img

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
    queue_query = """select TOP 1 * from TestMatchingConfig where Status = 'Pending'"""
    cur.execute(queue_query)
    queue_row_count = len(cur.fetchall())
    if queue_row_count == 0:
        time.sleep(idle_sleeptime)
    else:
        try:
            cur.execute(queue_query)
            queue_columns = [column[0] for column in cur.description]
            queue_row = cur.fetchone()
            queue_data = dict(zip(queue_columns, queue_row))
            test_games = queue_data['Games'].split()
            print(f"Process Start for Games {queue_data['Games']} on {queue_data['SettingName']}")
            matched_count = []
            total_count = []
            cur.execute("Update TestMatchingConfig set ProcessStartOn = CURRENT_TIMESTAMP where TestMatchingConfigId = ?", (queue_data['TestMatchingConfigId'],))
            # This will fetch the games which have status pending
            for test_game in test_games:
                key_query = "select * from cdp2000.WSA.AllGames where GameNumber = ?"
                cur.execute(key_query, (test_game,))
                key_columns = [column[0] for column in cur.description]
                key_row = cur.fetchone()
                key_data = dict(zip(key_columns, key_row))
                roster_query = """
                    SELECT * FROM CDPMediaCapture.dbo.Constellation 
                    WHERE PhotoType = 'Studio' AND TournamentID = ? AND MediaType = '.jpg' AND TeamKeys IN (?, ?) AND PhotoUse = ''"""
                cur.execute(roster_query, (queue_data['TournamentID'], str(key_data["HomeTeamKey"]), str(key_data["VisitorTeamKey"])))
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
                            WHERE RosterID = ? AND TournamentID = ? AND SBoundingBox is Not Null
                        """, (roster_data["RosterID"], roster_data["TournamentID"]))
                        result = cur.fetchone()
                        if result[0] > 0:
                            pass
                        else:
                            cur.execute("""
                                INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID, SBoundingBox)
                                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                            """, (roster_data["RosterID"], roster_data["TournamentID"], embedding.tobytes(), studio_image_path, roster_data["TeamKeys"], roster_data["ConstellationID"], studio_coordinates))
                            cnxn.commit()
                # Below query checks if there are already game photo embeddings exist in the GamePhotoEmbedding table, if found, it deletes them.
                cur.execute("DELETE FROM GamePhotoDetail WHERE GameNumber = ?", (test_game,))
                cur.execute("DELETE FROM GamePhotoEmbedding WHERE GameNumber = ?", (test_game,))
                game_query = """SELECT * FROM CDPMediaCapture.dbo.Constellation WHERE MediaType = '.jpg' AND GameNumber = ?"""
                cur.execute(game_query, (test_game,))
                game_columns = [column[0] for column in cur.description]
                game_rows = cur.fetchall()
                total_game_images = len(game_rows)
                # For each game photo, finding the number of faces and adding the details in the game photo detail table
                game_photos = 0
                for game_row in tqdm(game_rows, desc="Processing Game Images", unit="image", total=total_game_images):
                    game_data = dict(zip(game_columns, game_row))
                    game_image_path = game_data['KioskHiresFile'].replace(db_prefix_path, local_image_path)
                    game_image = Image.open(game_image_path)
                    gwidth, gheight = game_image.size
                    photo_embeddings = detect_align_embed_game_faces(game_image_path, finetuned_mtcnn, margin=margin)
                    id = cur.execute("""
                        INSERT INTO GamePhotoDetail (NuFaceDetected, ImagePath, RosterIDs, GameNumber, TournamentID, TournamentWeek, TournamentWeekNumber, GameDay, GameTime, GameField, GameDayNumber, GameTimeNumber, GameType, U, B, R, ConstellationID, IsMatch, IsGroupPhoto, IsTagged, IsAttempted, UpdatedOn, CreatedOn, HiresWidth, HiresHeight) OUTPUT inserted.GamePhotoDetailID
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                    """, (len(photo_embeddings), game_image_path, None, game_data["GameNumber"], game_data["TournamentID"], game_data["TournamentWeek"], game_data["TournamentWeekNumber"], game_data["GameDay"], game_data["GameTime"], game_data["GameField"], game_data["GameDayNumber"], game_data["GameTimeNumber"], game_data["GameType"], game_data["U"], game_data["B"], game_data["R"], game_data["ConstellationID"], gwidth, gheight)).fetchval()
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
                            INSERT INTO GamePhotoEmbedding (GamePhotoDetailID, GFaceEmbedding, UpdatedOn, CreatedOn, GameNumber, GBoundingBox)
                            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                        """, (id, embedding.tobytes(), game_data['GameNumber'], game_coordinates))
                        cnxn.commit()
                    # # Jersey number detection
                    # if config_data['MatchByJersey'] == 1:
                    #     dir_name, file_name = os.path.split(game_image_path)
                    #     new_file_name = f"sm_{file_name}"
                    #     gfinal_path = os.path.join(dir_name, new_file_name)
                    #     reader = easyocr.Reader(['en'], gpu=False)
                    #     #jimage = cv2.imread(gfinal_path)  
                    #     #gray_image = cv2.cvtColor(jimage, cv2.COLOR_BGR2GRAY)
                    #     jersey_result = reader.readtext(gfinal_path, allowlist ='0123456789')
                    #     try:
                    #         # output_img = draw_bounding_boxes(gfinal_path, result)
                    #         # # Convert BGR to RGB for matplotlib
                    #         # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                    #         # plt.imshow(output_img)
                    #         # plt.axis('off') 
                    #         # plt.show()
                    #         for jersey_detection in jersey_result:
                    #             text = jersey_detection[1]
                    #             confidence = jersey_detection[2]
                    #             if len(text)<3 and confidence>0.90:
                    #                 cur.execute("""UPDATE GamePhotoDetail SET JerseyNumbers = COALESCE(JerseyNumbers + ', ', '') + ?  WHERE ImagePath = ?""", (text, game_image_path))
                    #                 cnxn.commit()  
                    #                 #print("Database entry DONE!")
                    #             else:
                    #                 continue
                        # except ValueError as e:
                        #     print("Error:", e)
                cur.execute("""SELECT PlayerEmbeddingId, SFaceEmbeddings, RosterID FROM PlayerPhotoEmbedding WHERE TeamKey IN (?, ?) AND SFaceEmbeddings is Not Null""", (str(key_data["HomeTeamKey"]), str(key_data["VisitorTeamKey"])))
                player_columns = [column[0] for column in cur.description]
                player_embeddings = [dict(zip(player_columns, row)) for row in cur.fetchall()]
                # Process each studio embedding
                match_count = 0
                # Threshold value is taken directly from the MatchingConfig table from AI_Face_Recognition database
                euclidean_threshold = 1.5 - (float(queue_data['Threshold']) / 100) * (1.5 - 0.5)
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
                matched_count.append(match_count)
                total_count.append(game_photos)
            matched = sum(matched_count)
            total = sum(total_count)
            percentage = int((matched / total) * 100)
            print(f"Result for the Games {queue_data['Games']} on {queue_data['SettingName']}: ", percentage)
            cur.execute("""
                        UPDATE TestMatchingConfig
                        SET Status = 'Completed',
                            Result = ? 
                        WHERE TestMatchingConfigId = ?
                    """, (percentage, queue_data['TestMatchingConfigId']))
            cnxn.commit()
            if error_occurred is False:
                cur.execute("Update TestMatchingConfig set ProcessEndOn = CURRENT_TIMESTAMP where TestMatchingConfigId = ?", (queue_data['TestMatchingConfigId'],))
                cnxn.commit()

        except Exception as e:
            print(e)
            print("Error occurred at line:", traceback.extract_tb(e.__traceback__)[-1].lineno)
            break
        