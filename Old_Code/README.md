Project Title: AI_Face_Jersey_Matching

Introduction: 
During the tournament, at CooperstownDreamsPark, they click candid pictures of the players in every match. They upload the same on their website. When a player login into the website they can select the match they played in and checkout the images of themselves. Then they can buy the image digitally or with a frame from the venue. The issue in the above scenario is that all the players need to scroll through all the matches they played in and find their pictures. Using this code, we are trying to make that process automatic. When a player login to their account and select the match, using our script only that player's matching photos will show up. Apart from this we have also implemented the JerseyNumber detection, there are several images where the jersey number is visible in the image, we are matching that jersey number with the player. 

Library installation:
torch - 2.2.2
numpy - 1.26.4
image - 1.5.33
pillow - 10.2.0
opencv-python - 4.10.0.84
pyodbc - 5.2.0
os-sys - 
tqdm - 4.66.5
easyocr - 1.7.2
matplotlib - 3.9.2
facenet-pytorch - 2.6.0

Features: The code includes so many unique features listed below,
- Since the studio photos are repetitive the code makes sure that for a particular roster we store the embedding once.
- The code ensures that the GamePhotoDetail and GameEmbedding tables have the unique information every time by deleting the prior rows of any respective game before proceeding further, which reduces the chances of collision of embeddings and repetition.
- If there is any photo missing in the local system for any game, the process will not stop, it will set the status as 'error'.
- For any image, if the box is not found or gets eliminated by the filtered condition, in both the cases, the null row for the image will be added to their respective tables.
- If any other error occurs anywhere in the code the status of the game will be set as 'error'.
- If the status is InProgress not completed then code will make sure to wait for 20 minutes before proceeding with that game.
- RetryCount ensures that the error is being solved and same game is not coming for the process again and again.
- Once integrated this script with the process, it doesn't need any human intervention and can go on for the face matching task as long as there are photos available.
- Using the Status, RetryCount and Timestamp, one can easily figure out if the process is going properly or not!
Flow of code:

Functions:
1. alignment_procedure: It aligns the cropped face photos using eyes and nose coordinates we get from the MTCNN.
2. detect_align_embed_studio_faces: It loads the studio photos from the local, detect faces in them, align them using alignment procedure function and returns the embedding for the same.

Main Loop:
- While loop will work as long as it finds any game in the database. If not, it will take a break of 20 minutes. 
- Roster query takes the values of TeamKeys, takes studio photos from the Constellation table, takes the path, replace it with local path, loads the studio photos from local, goes into detect_align function, stores the embedding into the studio embedding table.
- Before proceeding further with the Game photos, we ensure that all the existing embedding and gamedetail have been deleted.
- Game query uses the Game number, gets the path from the Constellation table, replaces the path with the local one, goes in the align_detect_embed function, stores the embedding into the game embedding table.
- If MatcheByJersey = 1 in the MatchingConfig table, JerseyDetection code will be executed, and stores all the detected jerseynumbers at their respective locations.
- After the JerseyDetection task, Code picks one game, takes embedding of one studio image, tries matching it with all the game images from the respective game using the Euclidean distance, stores the rosterids of the matched ones in the table GamePhotoDetail.





