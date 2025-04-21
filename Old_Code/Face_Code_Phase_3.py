import numpy as np
import pyodbc as odbc


# Function to calculate Euclidean distance between embeddings
def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

threshold = 0.9


SERVER_NAME = '192.168.1.101'
DRIVER_NAME = 'ODBC Driver 17 for SQL Server'
DATABASE_NAME = 'AI_Face_Recognition'
cnxn = odbc.connect('DRIVER={ODBC Driver 17 for SQL Server}; \
                    SERVER='+SERVER_NAME+'; \
                    DATABASE = '+DATABASE_NAME+'; \
                    Uid=sa;Pwd=Masterly@123;')
cur = cnxn.cursor()

cur.execute("use AI_Face_Recognition")

#updated with the dictionary from the table
def compare_embeddings_and_update():
    """Compare SFaceEmbeddings with GFaceEmbedding and update MatchedFaces."""
    # Fetch all studio embeddings
    cur.execute("SELECT PlayerEmbeddingId, SFaceEmbeddings, GameNumber FROM PlayerPhotoEmbedding")
    player_columns = [column[0] for column in cur.description]
    player_embeddings = [dict(zip(player_columns, row)) for row in cur.fetchall()]
    global count 
    count = 0
    # Process each studio embedding
    for player in player_embeddings:
        count += 1
        s_embedding = np.frombuffer(player['SFaceEmbeddings'], dtype=np.float32).reshape(1, -1)
        # Fetch game embeddings with the same GameNumber
        cur.execute("""
            SELECT GamePhotoEmbeddingId, GFaceEmbedding, GamePhotoDetailId 
            FROM GamePhotoEmbedding 
            WHERE GameNumber = ?
        """, (player['GameNumber'],))
        game_columns = [column[0] for column in cur.description]
        game_embeddings = [dict(zip(game_columns, row)) for row in cur.fetchall()]
        # Compare with relevant game embeddings
        for game in game_embeddings:
            g_embedding = np.frombuffer(game['GFaceEmbedding'], dtype=np.float32).reshape(1, -1)
            distance = euclidean_distance(s_embedding, g_embedding)
            if distance < threshold:
                # Match found, increment MatchedFaces
                cur.execute("""
                    UPDATE GamePhotoDetail 
                    SET MatchedFaces = MatchedFaces + 1 
                    WHERE GamePhotoDetailId = ?
                """, (game['GamePhotoDetailId'],))
                cnxn.commit()
        print(f"Matching done for the Studio image {player['PlayerEmbeddingId']}")
        print(f"Remaining studio photos for matching: {len(player_embeddings) - count}")
        print("--------------------------------------------------------------------------------------------------------------------------------------------")

# Run the update process
compare_embeddings_and_update()
print("Matching task complete for all the studio images.")



# This iscompletely working code which compares the studio embedding with all the game embeddings
    # # Fetch all studio embeddings and their associated PlayerEmbeddingsId
    # cur.execute("SELECT PlayerEmbeddingId, SFaceEmbeddings, GameNumber FROM PlayerPhotoEmbedding")
    # player_columns = [column[0] for column in cur.description]
    # player_embeddings = [dict(zip(player_columns, player_row)) for player_row in cur.fetchall()]
    # # Fetch all game embeddings and their associated details
    # cur.execute("SELECT GamePhotoEmbeddingId, GFaceEmbedding, GamePhotoDetailId FROM GamePhotoEmbedding")
    # game_columns = [column[0] for column in cur.description]
    # game_embeddings = [dict(zip(game_columns, game_row)) for game_row in cur.fetchall()]

    # # Process each player embedding
    # count = 0
    # for player in player_embeddings:
    #     s_embeddings = np.frombuffer(player['SFaceEmbeddings'], dtype=np.float32).reshape(1, -1)

    #     # Compare with each game embedding
    #     for game in game_embeddings:
    #         g_embedding = np.frombuffer(game['GFaceEmbedding'], dtype=np.float32).reshape(1, -1)
    #         distance = euclidean_distance(s_embeddings, g_embedding)
    #         print(f"{game['GamePhotoEmbeddingId']}, {distance}")

    #         if distance < threshold:
    #             print("photo matched")
    #             count += 1
    #             print(count)
    #             # Match found, increment MatchedFaces
    #             cur.execute("""
    #                 UPDATE GamePhotoDetail 
    #                 SET MatchedFaces = MatchedFaces + 1 
    #                 WHERE GamePhotoDetailId = ?
    #             """, (game['GamePhotoDetailId'],))
    #             cnxn.commit()