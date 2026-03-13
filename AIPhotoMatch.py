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
import traceback
import sys
import settings
from tqdm import tqdm
#import time
#import easyocr
#from matplotlib import pyplot as plt
#import logging
#from logger_setup import init_logger
#init_logger()
# Suppress logging messages from EasyOCR
# logging.getLogger("easyocr").setLevel(logging.ERROR)

# Database Credentials for AI Face Matching
SERVER_NAME_DPP = settings.SERVER_NAME_DPP 
DRIVER_NAME_DPP = settings.DRIVER_NAME_DPP
DATABASE_NAME_DPP = settings.DATABASE_NAME_DPP
USER_NAME_DPP = settings.USER_NAME_DPP
PASSWORD_DPP = settings.PASSWORD_DPP

SERVER_NAME_CDPMC = settings.SERVER_NAME_CDPMC 
DRIVER_NAME_CDPMC = settings.DRIVER_NAME_CDPMC
DATABASE_NAME_CDPMC = settings.DATABASE_NAME_CDPMC
USER_NAME_CDPMC = settings.USER_NAME_CDPMC 
PASSWORD_CDPMC = settings.PASSWORD_CDPMC

SERVER_NAME_CDP2000 = settings.SERVER_NAME_CDP2000 
DRIVER_NAME_CDP2000 = settings.DRIVER_NAME_CDP2000
DATABASE_NAME_CDP2000 = settings.DATABASE_NAME_CDP2000
USER_NAME_CDP2000 = settings.USER_NAME_CDP2000 
PASSWORD_CDP2000 = settings.PASSWORD_CDP2000

# Parameters 
local_image_path = settings.LOCAL_IMAGE_PATH                     # Where your photos are stored locally
db_prefix_path = settings.DB_PREFIX_PATH                         # Will be replaced with the database path
min_face_size = settings.MIN_FACE_SIZE                           # Minimum faces you are allowing for the process eg. 30 (30*30 pixels)
idle_sleeptime = settings.IDLE_SLEEP_TIME                        # How many seconds do you want the while to sleep when there is no row found in the Queue data
processing_sleeptime = settings.PROCESSING_SLEEP_TIME            # How many seconds do you want the while to sleep when one game's process ends and another game's starts
retry_count = settings.RETRY_COUNT                               # How many number of tries are you allowing to each game when they face any error
timestamp = settings.PROCESSING_TIMESTAMP                        # If any game has status InProgress, after how many seconds do you want to give that game a try
margin = settings.MARGIN                                         # How much more of the face do you want in the studio images eg. 80 (from each side it will take 80 more pixels)
gamesperscheduler = settings.GAMES_PER_SCHEDULER                 # How many games do you want to process in one go

class FinetunedMTCNN(MTCNN):
#(self, image_size=160, margin=5, **kwargs): # use this after some time to improve the final results
    def __init__(self, **kwargs):
        super(FinetunedMTCNN, self).__init__(**kwargs)
        self.device = kwargs.get('device', torch.device('cpu'))
        # Create your custom, finetuned P-Net, R-Net, O-Net here
        self.pnet = PNet().to(self.device)
        self.rnet = RNet().to(self.device)
        self.onet = ONet().to(self.device)
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


def detect_align_embed_studio_faces(spath, mtcnn_model, curDPP, connDPP, queue_data, roster_data, margin=0):
    """function takes a photo and provides embedding for the same."""
    global error_occurred, studio_coordinates
    cropped_photos = []
    # Check if the file exists at the given path
    if not os.path.exists(spath):
        error_occurred = True
        return cropped_photos 
    try:
        image = Image.open(spath)
        boxes, confidences, landmarks = mtcnn_model.detect(image, landmarks=True)
        global photo_embeddings
        photo_embeddings = []
        if boxes is None:
            curDPP.execute("""
            SELECT COUNT(*) FROM PlayerPhotoEmbedding
            WHERE RosterID = ? AND TournamentID = ?
            """, (roster_data["RosterID"], roster_data["TournamentID"]))
            result = curDPP.fetchone()
            if result[0] > 0:
                pass
            else:
                curDPP.execute("""
                    INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID)
                    VALUES (?, ?, NULL, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """, (roster_data["RosterID"], roster_data["TournamentID"], spath, roster_data["TeamKeys"], roster_data["ConstellationID"]))
                connDPP.commit()
        else:
            threshold = settings.THRESHOLD_STUDIO  # Set a confidence threshold
            # Filter detected faces based on the confidence score
            filtered_faces = [i for i, confidence in enumerate(confidences) if confidence > threshold]
            if filtered_faces is None:
                curDPP.execute("""
                SELECT COUNT(*) FROM PlayerPhotoEmbedding
                WHERE RosterID = ? AND TournamentID = ?
                """, (roster_data["RosterID"], roster_data["TournamentID"]))
                result = curDPP.fetchone()
                if result[0] > 0:
                    pass
                else:
                    curDPP.execute("""
                    INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID)
                    VALUES (?, ?, NULL, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """, (roster_data["RosterID"], roster_data["TournamentID"], spath, roster_data["TeamKeys"], roster_data["ConstellationID"]))
                    connDPP.commit()
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
                    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0  # Convert to tensor, normalize
                    #show_image = Image.fromarray(face_image)
                    #show_image.show()
                    embedding = facenet(image)  # Get the embedding
                    embedding = normalize(embedding, p=2, dim=1)  # L2 normalization of embeddings
                    embedding = embedding.detach().cpu().numpy()
                    if embedding is not None:
                        photo_embeddings.append(embedding)  # Store the index and embedding
            return photo_embeddings
    except Exception as e:
        # Update game status to error
        print(f"Error in gamenumber: {queue_data['GameNumber']}",e)
        curDPP.execute("""UPDATE AITournamentQueue SET Status = 'error', RetryCount = RetryCount + 1 WHERE GameNumber = ?""", (queue_data["GameNumber"],))
        connDPP.commit()
    return cropped_photos





def detect_align_embed_game_faces(gpath, mtcnn_model, curDPP, connDPP, queue_data, margin=0):
    """function takes a photo and provides embedding for the same."""
    global error_occurred, isgamephotonull, game_coordinates
    cropped_photos = []
    # Check if the file exists at the given path
    if not os.path.exists(gpath):
        error_occurred = True
        return cropped_photos 
    try:
        image = Image.open(gpath)
        boxes, confidences, landmarks = mtcnn_model.detect(image, landmarks=True)
        global photo_embeddings
        photo_embeddings = []
        if boxes is None:
            isgamephotonull = True
        else:
            # Set a confidence threshold
            threshold = settings.THRESHOLD_GAME
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
                    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0  # Convert to tensor, normalize
                    embedding = facenet(image)  # Get the embedding
                    embedding = normalize(embedding, p=2, dim=1)  # L2 normalization of embeddings
                    embedding = embedding.detach().cpu().numpy()
                    if embedding is not None:
                        photo_embeddings.append(embedding)  # Store the index and embedding
            return photo_embeddings
    except Exception as e:
        # Update game status to error
        print(f"Error in gamenumber: {queue_data['GameNumber']}",e)
        curDPP.execute("""UPDATE AITournamentQueue SET Status = 'error', RetryCount = RetryCount + 1 WHERE GameNumber = ?""", (queue_data["GameNumber"],))
        connDPP.commit()
    return cropped_photos  

def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)
# TO draw the box over the image
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

def process_constellation_updates(curDPP, curCDPMC, curCDP2000, connDPP):
    last_processed_id = None
    latest_constellation_id = None
    game_info_list = []
    try:
        # Step 1: Get last processed constellation ID
        curDPP.execute("select top 1 ConstellationID from ConstellationIDInfo order by 1 desc")
        last_processed = curDPP.fetchone()
        last_processed_id = last_processed.ConstellationID

        # Step 2: Get latest ConstellationID
        curCDPMC.execute("SELECT TOP 1 ConstellationID FROM CDPMediaCapture.dbo.Constellation ORDER BY ConstellationID DESC")
        latest = curCDPMC.fetchone()
        latest_constellation_id = latest.ConstellationID 
        print("Latest ConstellationID: ", latest_constellation_id)

        # Step 3: Get distinct GameNumbers for new entries
        curCDPMC.execute("""
            SELECT DISTINCT GameNumber, TournamentID
            FROM Constellation
            WHERE GameNumber IS NOT NULL AND ConstellationID > ?
            ORDER BY GameNumber DESC
        """, last_processed_id)
        game_info_list = curCDPMC.fetchall() 

        # Step 4: Insert each GameNumber into AITournamentQueue
        for game_number, tournament_id in game_info_list:
            curCDP2000.execute("""
                    SELECT TournamentID, GameNumber, HomeTeamKey, VisitorTeamKey
                    FROM CDP2000.WSA.AllGames
                    WHERE GameNumber = ? and TournamentID = ?
            """, game_number, tournament_id)
            keyrows = curCDP2000.fetchall()

            for row in keyrows:
                # Check if the record exists
                curDPP.execute("""
                    SELECT COUNT(1) 
                    FROM AITournamentQueue 
                    WHERE TournamentID = ? AND GameNumber = ?
                """, row.TournamentID, row.GameNumber)
                
                exists = curDPP.fetchone()[0]
                
                if exists:
                    # Update the existing record
                    curDPP.execute("""
                        UPDATE AITournamentQueue 
                        SET Status = 'pending', RetryCount = 0, UpdatedOn = CURRENT_TIMESTAMP
                        WHERE TournamentID = ? AND GameNumber = ?
                    """, row.TournamentID, row.GameNumber)
                else:
                    # Insert new record
                    curDPP.execute("""
                        INSERT INTO AITournamentQueue (TournamentID, GameNumber, TeamKey1, TeamKey2, Status, RetryCount)
                        VALUES (?, ?, ?, ?, 'pending', 0)
                    """, row.TournamentID, row.GameNumber, row.HomeTeamKey, row.VisitorTeamKey)
            connDPP.commit()


        # Step 5: Update ConstellationIDInfo table
        if latest_constellation_id is not None:
            curDPP.execute("""INSERT INTO ConstellationIDInfo (ConstellationID, CreatedOn)
                    VALUES (?, CURRENT_TIMESTAMP)
            """, latest_constellation_id)
            connDPP.commit()

    except Exception as e:
        connDPP.rollback()
        print(f"Error in process_constellation_updates function: {e}")
        traceback.print_exc() 
        _, _, tb = sys.exc_info()
        line_number = tb.tb_lineno
        print(f"Error occurred on line: {line_number}")
    return last_processed_id, latest_constellation_id, game_info_list

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Initializing the Facenet model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device) 

# Initializing the MTCNN model
finetuned_mtcnn = FinetunedMTCNN(keep_all=True, device=device, min_face_size=min_face_size)


# Main service starts from here
isgamephotonull = False
error_occurred = False
def process_tasks_once():
    #DB name: DREAMSPARKPHOTOS -> DPP
    connDPP = odbc.connect('DRIVER='+DRIVER_NAME_DPP+'; \
                        SERVER='+SERVER_NAME_DPP+'; \
                        DATABASE = '+DATABASE_NAME_DPP+'; \
                        Uid='+USER_NAME_DPP+';Pwd='+PASSWORD_DPP+';')
    curDPP = connDPP.cursor() 
    curDPP.execute("use DREAMSPARKPHOTOS") 

    #DB name: CDPMediaCapture -> CDPMC
    connCDPMC = odbc.connect('DRIVER='+DRIVER_NAME_CDPMC+'; \
                        SERVER='+SERVER_NAME_CDPMC+'; \
                        DATABASE = '+DATABASE_NAME_CDPMC+'; \
                        Uid='+USER_NAME_CDPMC+';Pwd='+PASSWORD_CDPMC+';')
    curCDPMC = connCDPMC.cursor()
    curCDPMC.execute("use CDPMediaCapture")

    #DB name: CDP2000 -> CDP2000
    connCDP2000 = odbc.connect('DRIVER='+DRIVER_NAME_CDP2000+'; \
                        SERVER='+SERVER_NAME_CDP2000+'; \
                        DATABASE = '+DATABASE_NAME_CDP2000+'; \
                        Uid='+USER_NAME_CDP2000+';Pwd='+PASSWORD_CDP2000+';')
    curCDP2000 = connCDP2000.cursor()
    curCDP2000.execute("use CDP2000")
    try:
        start_constellationid, end_constellationid, game_info = process_constellation_updates(curDPP, curCDPMC, curCDP2000, connDPP)
        constgames = [i[0] for i in game_info]
        print(f"Face Matching service initiated for games between ConstellationID {start_constellationid} and {end_constellationid}")
        print("Number of games identified from constellation: ", len(constgames))
        print("New images identified for GameNumbers: ", constgames)
        queue_query = f"""SELECT TOP {gamesperscheduler} * FROM AITournamentQueue WHERE (Status IN('pending','error') AND RetryCount < ?) 
                            OR (Status IN ('InProgress') AND DATEDIFF(SECOND, UpdatedOn, CURRENT_TIMESTAMP) > ?)"""
        curDPP.execute(queue_query, (retry_count, timestamp))
        queue_columns = [column[0] for column in curDPP.description]
        queue_rows = curDPP.fetchall()
        game_numbers = [row[queue_columns.index('GameNumber')] for row in queue_rows]
        print("Number of games picked from queue: ", len(game_numbers))
        print("Processing started for games: ", game_numbers)
        if game_numbers:
            placeholders = ','.join('?' for _ in game_numbers)
            curDPP.execute(f"""
                UPDATE AITournamentQueue
                SET Status = 'InProgress', UpdatedOn = CURRENT_TIMESTAMP
                WHERE GameNumber IN ({placeholders})
            """, game_numbers)
            connDPP.commit()
        for queue_row in queue_rows:
            queue_data = dict(zip(queue_columns, queue_row))
            global game_error_occurred
            game_error_occurred = queue_data["GameNumber"] #to use the gamenumber outside of the try block
            curDPP.execute("""UPDATE AITournamentQueue 
                        SET ProcessStartOn = CURRENT_TIMESTAMP, UpdatedOn = CURRENT_TIMESTAMP, Status = 'InProgress'
                            WHERE GameNumber = ?""", (queue_data["GameNumber"],))
            connDPP.commit()
            print(f"Photo matching process begins for Game Number: {queue_data['GameNumber']}")

            #Fetch the Team details 
            curCDP2000.execute("SELECT * FROM Team WHERE TournamentID =? and TeamKey in (?,?)",(queue_data['TournamentID'], queue_data['TeamKey1'], queue_data['TeamKey2']))
            gameTeam_column = [column[0] for column in curCDP2000.description]
            gameTeam_rows = curCDP2000.fetchall()
            gameTeam_data = [dict(zip(gameTeam_column, row)) for row in gameTeam_rows]


            #Fetch the roster details
            curCDP2000.execute("select * from roster where TeamKey in (?,?)",(queue_data['TeamKey1'], queue_data['TeamKey2']))
            team_roster_column = [column[0] for column in curCDP2000.description]
            team_roster_rows =curCDP2000.fetchall()
            teamRosters = [dict(zip(team_roster_column, row)) for row in team_roster_rows]

            # Below query will return the studio photos for the particular game
            roster_query = """SELECT * FROM Constellation WHERE PhotoType = 'Studio' AND TournamentID = ? AND MediaType = '.jpg' AND TeamKeys IN (?, ?) AND PhotoUse = ''"""
            curCDPMC.execute(roster_query, (queue_data['TournamentID'], queue_data['TeamKey1'], queue_data['TeamKey2']))
            roster_columns = [column[0] for column in curCDPMC.description]
            roster_rows = curCDPMC.fetchall()
            roster_team_data = [dict(zip(roster_columns, row)) for row in roster_rows]
            total_studio_images = len(roster_rows)
            print(f"Studio photos embedding task initiated for {total_studio_images} images for Game Number: {queue_data['GameNumber']}")
            roster_ids = [int(row['RosterID']) for row in roster_team_data if int(row['RosterID'])]
            placeholders = ', '.join(['?'] * len(roster_ids))
            curDPP.execute(f"SELECT RosterID FROM PlayerPhotoEmbedding WHERE RosterID IN ({placeholders})", roster_ids)
            # for each studio photos, we will find embedding and adding it to the photoembedding table
            embedded_roster_ids = [row[0] for row in curDPP.fetchall()]

            #Create Filtered row which not have the embedings in file
            filtered_roster_rows = [
                row for row in roster_rows
                if row[roster_columns.index('RosterID')] and int(row[roster_columns.index('RosterID')]) not in embedded_roster_ids
            ]
            
            studio_photos = 0
            for roster_row in tqdm(filtered_roster_rows, desc="Processing Studio Images", unit="image", total=len(filtered_roster_rows), disable=True):
                roster_data = dict(zip(roster_columns, roster_row))

                if not roster_data.get('RosterID'):
                    continue

                if int(roster_data['RosterID']) in embedded_roster_ids:
                    continue

                studio_image_path = roster_data['KioskHiresFile']#.replace(db_prefix_path, local_image_path)
                if studio_image_path is None:
                    print(f"Studio Imagepath is missing for rosterid: {roster_data['RosterID']}, GameNumber: {queue_data['GameNumber']}")
                    curDPP.execute("""
                        INSERT INTO MissingImagePath (ImagePath, GameNumber, TournamentID, CreatedOn, ConstellationID, Status, TeamKey, RosterID)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, 'pending', ?, ?)
                    """, (studio_image_path, queue_data['GameNumber'], roster_data["TournamentID"], roster_data["ConstellationID"], roster_data["TeamKeys"], roster_data["RosterID"]))
                    connDPP.commit()
                    continue
                sphoto_embeddings = detect_align_embed_studio_faces(studio_image_path, finetuned_mtcnn, curDPP, connDPP, queue_data, roster_data, margin=margin)
                # Checking if the embedding is already present or not for the specific studio photo
                for embedding in sphoto_embeddings:
                    studio_photos+=1
                    curDPP.execute("""
                        INSERT INTO PlayerPhotoEmbedding (RosterID, TournamentID, SFaceEmbeddings, ImagePath, TeamKey, UpdatedOn, CreatedOn, ConstellationID, SBoundingBox)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                    """, (roster_data["RosterID"], roster_data["TournamentID"], embedding.tobytes(), studio_image_path, roster_data["TeamKeys"], roster_data["ConstellationID"], studio_coordinates))
                    connDPP.commit()
            print(f"Studio photos embedding task completed for {studio_photos} images for  Game Number: {queue_data['GameNumber']}")
            excludeConstillationsList = curDPP.execute("""select constellationID from AIResult where TournamentID = ? and GameNumber = ? and Manual =1 and IsGroup = 0""", (queue_data["TournamentID"],queue_data["GameNumber"]))
            excludeConstillationIds = [row[0] for row in excludeConstillationsList.fetchall()]
            excludeConstillationIdsString = ','.join(['?'] * len(excludeConstillationIds))


            # Below query checks if there are already game photo embeddings exist in the GamePhotoEmbedding table, if found, it deletes them.
            if excludeConstillationIds:
                curDPP.execute(f"DELETE FROM GamePhotoDetail WHERE GameNumber = ? and constellationID NOT IN ({excludeConstillationIdsString})", (queue_data["GameNumber"],*excludeConstillationIds))
                curDPP.execute(f"DELETE FROM GamePhotoEmbedding WHERE GameNumber = ? and constellationID NOT IN ({excludeConstillationIdsString})", (queue_data["GameNumber"],*excludeConstillationIds))
            else:
                curDPP.execute("DELETE FROM GamePhotoDetail WHERE GameNumber = ?", (queue_data["GameNumber"],))
                curDPP.execute("DELETE FROM GamePhotoEmbedding WHERE GameNumber = ?", (queue_data["GameNumber"],))
            curDPP.execute("DELETE FROM AIResult WHERE Manual = 0 AND GameNumber = ?", (queue_data["GameNumber"],))
            # Below query will provide threshold value and Jersey number detection flag
            config_query = """SELECT TOP 1 * FROM MatchingConfig where IsActive = 1"""
            curDPP.execute(config_query)
            config_columns = [column[0] for column in curDPP.description]
            config_row = curDPP.fetchone()
            config_data = dict(zip(config_columns, config_row))
            if excludeConstillationIds:
                game_query = f"""SELECT * FROM Constellation WHERE MediaType = '.jpg' AND GameNumber = ? and constellationID NOT IN ({excludeConstillationIdsString})"""
                curCDPMC.execute(game_query, (queue_data["GameNumber"],*excludeConstillationIds))
            else:
                game_query = """SELECT * FROM Constellation WHERE MediaType = '.jpg' AND GameNumber = ?"""
                curCDPMC.execute(game_query, (queue_data["GameNumber"],))
            game_columns = [column[0] for column in curCDPMC.description]
            game_rows = curCDPMC.fetchall()
            total_game_images = len(game_rows)
            print(f"Game photos embedding and matching task initiated for {total_game_images} game images for Game Number: {queue_data['GameNumber']}")

            #Fetch player Details with embadings
            curDPP.execute("""SELECT PlayerEmbeddingId, SFaceEmbeddings, RosterID, TeamKey FROM PlayerPhotoEmbedding WHERE TeamKey IN (?, ?) AND SFaceEmbeddings is Not Null""", (queue_data['TeamKey1'], queue_data['TeamKey2']))
            player_columns = [column[0] for column in curDPP.description]
            player_embeddings = [dict(zip(player_columns, row)) for row in curDPP.fetchall()]
            # Threshold value is taken directly from the MatchingConfig table from AIPhotoMatch database
            euclidean_threshold = settings.DEFAULT_EUCLIDEAN_THRESHOLD_MAX - (float(config_data['Threshold']) / 100) * (settings.DEFAULT_EUCLIDEAN_THRESHOLD_MAX - settings.DEFAULT_EUCLIDEAN_THRESHOLD_MIN)


            game_photos = 0
            for game_row in tqdm(game_rows, desc="Processing Game Images", unit="image", total=total_game_images, disable=True):
                game_data = dict(zip(game_columns, game_row))
                game_image_path = game_data['KioskHiresFile']#.replace(db_prefix_path, local_image_path)
                if game_image_path is None:
                    curDPP.execute("""
                        INSERT INTO MissingImagePath (ImagePath, GameNumber, TournamentID, CreatedOn, ConstellationID, Status, TeamKey, RosterID)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, 'pending', ?, NULL)
                    """, (game_image_path, queue_data['GameNumber'], game_data["TournamentID"], game_data["ConstellationID"], game_data["TeamKeys"]))
                    connDPP.commit()
                    continue
                gphoto_embeddings = detect_align_embed_game_faces(game_image_path, finetuned_mtcnn, curDPP, connDPP, queue_data, margin=margin)
                id = curDPP.execute("""
                    INSERT INTO GamePhotoDetail (NuFaceDetected, ImagePath, RosterIDs, GameNumber, TournamentID, TournamentWeek, TournamentWeekNumber, GameDay, GameTime, GameField, GameDayNumber, GameTimeNumber, GameType, U, B, R, ConstellationID, IsMatch, IsGroupPhoto, IsTagged, IsAttempted, UpdatedOn, CreatedOn, HiresWidth, HiresHeight) OUTPUT inserted.GamePhotoDetailID
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                """, (len(gphoto_embeddings), game_image_path, None, game_data["GameNumber"], game_data["TournamentID"], game_data["TournamentWeek"], game_data["TournamentWeekNumber"], game_data["GameDay"], game_data["GameTime"], game_data["GameField"], game_data["GameDayNumber"], game_data["GameTimeNumber"], game_data["GameType"], game_data["U"], game_data["B"], game_data["R"], game_data["ConstellationID"],1 if len(gphoto_embeddings) > 1 else 0 , 0, 0)).fetchval()
                connDPP.commit()

            
                if len(gphoto_embeddings) == 0:
                    curDPP.execute("""
                        INSERT INTO GamePhotoEmbedding (GamePhotoDetailID, GFaceEmbedding, UpdatedOn, CreatedOn, GameNumber) VALUES (?, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)""", (id, game_data['GameNumber']))
                    connDPP.commit()
                    
                # For each game photo, adding the embedding and gamephotodetailid into the gamephotoembedding table
                for embedding in gphoto_embeddings:
                    game_photos += 1
                    curDPP.execute("""
                        INSERT INTO GamePhotoEmbedding (GamePhotoDetailID, GFaceEmbedding, GameNumber, GBoundingBox, CreatedOn, UpdatedOn)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (id, embedding.tobytes(), game_data['GameNumber'], game_coordinates))
                    connDPP.commit()
                    
                    isHomeTeamPhoto = game_data['R'] ==1
                    isVisitorTeamPhoto = game_data['B'] ==1
                    filteredPlayers = player_embeddings
                    filteredPlayers = [player_embedding for player_embedding in player_embeddings if str(player_embedding['TeamKey']) == str(queue_data['TeamKey1'])]

                    if isHomeTeamPhoto and isVisitorTeamPhoto:
                        pass  # keep all players
                    elif isHomeTeamPhoto:
                        filteredPlayers = [player_embedding for player_embedding in player_embeddings if str(player_embedding['TeamKey']) == str(queue_data['TeamKey1'])]
                    elif isVisitorTeamPhoto:
                        filteredPlayers = [player_embedding for player_embedding in player_embeddings if str(player_embedding['TeamKey']) == str(queue_data['TeamKey2'])]

                    for player in filteredPlayers:
                        player_team_details  =  next((team for team in gameTeam_data if team['TeamKey'] == player['TeamKey']), None)
                        teamName = (player_team_details or {}).get("TeamName", "")
                        teamNumber = (player_team_details or {}).get("TeamNumber", 0)
                        #Find player details
                        player_details =  next((roster for roster in teamRosters if roster['RosterID'] == player['RosterID']), None)
                        s_embedding = np.frombuffer(player['SFaceEmbeddings'], dtype=np.float32).reshape(1, -1)
                        distance = euclidean_distance(s_embedding, embedding)
                        if distance < euclidean_threshold:
                            curDPP.execute("update GamePhotoDetail set IsMatch = 1 where GamePhotoDetailId = ?", (id))
                            connDPP.commit()
                            curDPP.execute("SELECT TOP 1 * from AIResult where ConstellationID = ?  AND RosterID = ? AND GameNumber = ?", (game_data['ConstellationID'], player['RosterID'], game_data['GameNumber']))
                            columns = [desc[0] for desc in curDPP.description]
                            row = curDPP.fetchone()
                            aiRecordDetails = dict(zip(columns, row)) if row else None
                            #aiRecordDetails = curDPP.fetchone()
                            # Check the record already exist or manual overrided then no need to take action.
                            if aiRecordDetails is None:
                                # Row doesn't exist, so insert a new one                               
                                curDPP.execute("""INSERT INTO AIResult (ConstellationID, ImagePath, RosterID, NuFaces, IsGroup, GameNumber, TournamentID, U, B, R, CreatedOn,TeamKey,TeamNumber,TeamName,PlayerFirstName,PlayerLastName) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP,?,?,?,?,?)""", 
                                            (game_data['ConstellationID'], game_image_path, player['RosterID'], len(gphoto_embeddings), 1 if len(gphoto_embeddings) > 1 else 0, game_data['GameNumber'], game_data['TournamentID'], game_data['U'], game_data['B'],
                                            game_data['R'],player['TeamKey'],teamNumber,teamName,player_details['FirstName'],player_details['LastName']))
                                connDPP.commit()                    
            print(f"Game photos embedding and matching task completed for {game_photos} game photos for Game Number: {queue_data['GameNumber']}")
            print(f"Photo matching process completed for Game Number: {queue_data['GameNumber']}")
            print("----------------------------------------------------------------------------------------------------------------------------")
            global error_occurred
            if error_occurred is False:
                curDPP.execute("""UPDATE AITournamentQueue SET Status = 'completed', ProcessEndOn = CURRENT_TIMESTAMP WHERE GameNumber = ?""", (queue_data['GameNumber'],))
                connDPP.commit()
            else:
                curDPP.execute("""UPDATE AITournamentQueue SET Status = 'error', ProcessEndOn = CURRENT_TIMESTAMP, RetryCount = RetryCount + 1 WHERE GameNumber = ?""", (queue_data['GameNumber'],))
                connDPP.commit()
                error_occurred = False
            #time.sleep(processing_sleeptime)

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()  # This prints the full traceback including line number
        # If you want to manually access the line number:
        _, _, tb = sys.exc_info()
        line_number = tb.tb_lineno
        print(f"Error occurred on line: {line_number} in Game Number: {game_error_occurred}")
        curDPP.execute("""UPDATE AITournamentQueue SET Status = 'error', ProcessEndOn = CURRENT_TIMESTAMP, RetryCount = RetryCount + 1 WHERE GameNumber = ?""", (game_error_occurred,))
        connDPP.commit()
        
    finally:
        # Close DB connections safely
        try:
            if curDPP: curDPP.close()
            if connDPP: connDPP.close()
        except: pass

        try:
            if curCDPMC: curCDPMC.close()
            if connCDPMC: connCDPMC.close()
        except: pass

        try:
            if curCDP2000: curCDP2000.close()
            if connCDP2000: connCDP2000.close()
        except: pass