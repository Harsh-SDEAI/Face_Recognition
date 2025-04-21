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


# Database Credentials
SERVER_NAME = '192.168.1.101' 
DRIVER_NAME = 'ODBC Driver 17 for SQL Server'
DATABASE_NAME = 'AI_Data_Creation'
USER_NAME = 'sa' 
PASSWORD = 'Masterly@123'

# Parameters 
local_image_path = 'E:'                             # Where your photos are stored locally
db_prefix_path = '\\\\172.16.17.136\\PHOTO_ROOT'    # Will be replaced with the database path
                                  # Matching threshold (Distances less than threshold will get picked as matched)


# Connection to Database:
cnxn = odbc.connect('DRIVER='+DRIVER_NAME+'; \
                    SERVER='+SERVER_NAME+'; \
                    DATABASE = '+DATABASE_NAME+'; \
                    Uid='+USER_NAME+';Pwd='+PASSWORD+';')
cur = cnxn.cursor()
cur.execute("use AI_Data_Creation") 

def draw_bounding_boxes(img_path, results):
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at path: {img_path}")
    # Set font and spacer for text placement
    font = cv2.FONT_HERSHEY_SIMPLEX
    spacer = 20  # Initial spacer for vertical text placement
    # Ensure results is a list (even if only one detection is provided)
    if not isinstance(results, list):
        raise ValueError("Results should be a list of detections.")
    # Loop through detections
    for detection in results:
        # Ensure detection has three elements: bounding box, text, and confidence
        if len(detection) != 3:
            raise ValueError("Each detection should have bounding box, text, and confidence.")
        # Extract bounding box, text, and confidence
        bbox = detection[0]
        text = detection[1]
        confidence = detection[2]
        # Ensure bounding box has 4 points
        if len(bbox) != 4:
            raise ValueError("Bounding box should have 4 points.")
        # Extract coordinates for rectangle
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        # Draw rectangle
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        # Draw text with confidence
        display_text = f"{text} ({confidence:.2f})"
        img = cv2.putText(img, display_text, (top_left[0], top_left[1] - 10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        # Optionally display all text stacked below the image
        img = cv2.putText(img, display_text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        spacer += 20
    # Return the image with drawn bounding boxes
    return img

while True:
    queue_count_query = "select * from DataQueue where Status in ('pending', 'error', 'InProgress')"
    cur.execute(queue_count_query)
    queue_count = len(cur.fetchall())
    print("Number of Games in Queue: ", queue_count)
    queue_query = "SELECT TOP 1 * FROM DataQueue WHERE Status IN('pending', 'error')"
    cur.execute(queue_query)
    queue_row_count = len(cur.fetchall())
    cur.execute(queue_query)
    queue_columns = [column[0] for column in cur.description]
    queue_rows = cur.fetchall()
    # This will fetch the games which have status pending
    for queue_row in queue_rows:
        queue_data = dict(zip(queue_columns, queue_row))
        cur.execute("""UPDATE DataQueue 
                    SET Status = 'InProgress' WHERE GameNumber = ?""", (queue_data["GameNumber"],))
        cnxn.commit()
        print(f"Photo matching process begins for Game Number: {queue_data['GameNumber']}")
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
            dir_name, file_name = os.path.split(game_image_path)
            new_file_name = f"sm_{file_name}"
            gfinal_path = os.path.join(dir_name, new_file_name)
            reader = easyocr.Reader(['en'], gpu=False)
            result = reader.readtext(gfinal_path, allowlist ='0123456789')
            print(game_image_path)
            print(result)
            try:
                output_img = draw_bounding_boxes(gfinal_path, result)
                # Convert BGR to RGB for matplotlib
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                plt.imshow(output_img)
                plt.axis('off') 
                #plt.show()
                for detection in result:
                    text = detection[1]
                    confidence = detection[2]
                    if len(text)<3 and confidence>0.90:
                        cur.execute("""UPDATE GameDetail SET JerseyNumbers = COALESCE(JerseyNumbers + ', ', '') + ?  WHERE ImagePath = ?""", (text, game_image_path))
                        cnxn.commit()  
                        print("database entrY DONE!")
                    else:
                        continue
            except ValueError as e:
                print("Error:", e)
            










