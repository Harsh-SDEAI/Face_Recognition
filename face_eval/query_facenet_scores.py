"""Query best FaceNet cosine similarity scores per studio face per game.

Usage:
    python query_facenet_scores.py
    python query_facenet_scores.py --top 5 --game 28049

Pulls FaceNet embeddings from the DB, computes cosine similarity between
every studio face and every game face, and prints the top-N matches.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from utils.db import connect


def load_embeddings(model_name: str = "facenet"):
    conn = connect()

    studio = pd.read_sql("""
        SELECT e.FaceID, e.Embedding, sp.TeamKey, sp.ImagePath,
               f.Confidence, f.BoxX1, f.BoxY1, f.BoxX2, f.BoxY2
        FROM EvalEmbedding e
        JOIN EvalFaceDetection f ON f.FaceID = e.FaceID
        JOIN EvalStudioPhoto sp ON sp.StudioPhotoID = f.SourceID
        WHERE e.ModelName = ? AND f.SourceType = 'S'
    """, conn, params=[model_name])

    game = pd.read_sql("""
        SELECT e.FaceID, e.Embedding, gp.GameNumber, gp.ImagePath,
               f.Confidence, f.BoxX1, f.BoxY1, f.BoxX2, f.BoxY2
        FROM EvalEmbedding e
        JOIN EvalFaceDetection f ON f.FaceID = e.FaceID
        JOIN EvalGamePhoto gp ON gp.GamePhotoID = f.SourceID
        WHERE e.ModelName = ? AND f.SourceType = 'G'
    """, conn, params=[model_name])

    conn.close()
    return studio, game


def blob_to_vec(blob) -> np.ndarray:
    return np.frombuffer(bytes(blob), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3, help="Top-N matches per studio face")
    parser.add_argument("--game", type=int, default=None, help="Filter to a specific game number")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min cosine similarity to show")
    args = parser.parse_args()

    studio_df, game_df = load_embeddings("facenet")

    if studio_df.empty:
        print("No FaceNet studio embeddings found.  Run generate_embeddings.py first.")
        sys.exit(1)
    if game_df.empty:
        print("No FaceNet game embeddings found.")
        sys.exit(1)

    if args.game:
        game_df = game_df[game_df["GameNumber"] == args.game]
        if game_df.empty:
            print(f"No game embeddings for game {args.game}.")
            sys.exit(1)

    # Parse embeddings into numpy arrays
    studio_vecs = {int(r.FaceID): blob_to_vec(r.Embedding) for r in studio_df.itertuples()}
    game_vecs = {int(r.FaceID): blob_to_vec(r.Embedding) for r in game_df.itertuples()}

    # Build game face metadata lookup
    game_meta = {}
    for r in game_df.itertuples():
        game_meta[int(r.FaceID)] = {
            "GameNumber": int(r.GameNumber),
            "ImagePath": r.ImagePath,
            "Confidence": float(r.Confidence),
        }

    # Stack game vectors for batch cosine computation
    game_ids = list(game_vecs.keys())
    game_matrix = np.stack([game_vecs[gid] for gid in game_ids])  # (N_game, 512)

    games_in_scope = sorted(game_df["GameNumber"].unique())
    print(f"FaceNet similarity report")
    print(f"Studio faces: {len(studio_vecs)} | Game faces: {len(game_vecs)} | Games: {games_in_scope}")
    print(f"Showing top {args.top} matches per studio face (threshold >= {args.threshold})")
    print("=" * 90)

    for s_row in studio_df.itertuples():
        s_id = int(s_row.FaceID)
        s_vec = studio_vecs[s_id]
        s_path = s_row.ImagePath.split("\\")[-1] if "\\" in s_row.ImagePath else s_row.ImagePath.split("/")[-1]

        # Cosine similarity = dot product (both L2-normalized)
        sims = game_matrix @ s_vec  # (N_game,)

        # Get top-N
        top_indices = np.argsort(sims)[::-1][:args.top * 3]  # get extra then filter
        results = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim < args.threshold:
                break
            gid = game_ids[idx]
            meta = game_meta[gid]
            if args.game and meta["GameNumber"] != args.game:
                continue
            g_path = meta["ImagePath"].split("\\")[-1] if "\\" in meta["ImagePath"] else meta["ImagePath"].split("/")[-1]
            results.append({
                "GameFaceID": gid,
                "GameNumber": meta["GameNumber"],
                "Cosine": round(sim, 4),
                "Euclidean": round(float(np.sqrt(2 - 2 * sim)), 4),
                "GamePhoto": g_path,
                "DetConf": round(meta["Confidence"], 3),
            })
            if len(results) >= args.top:
                break

        print(f"\nStudio face {s_id} | Team {int(s_row.TeamKey)} | {s_path} | det_conf={s_row.Confidence:.3f}")
        if not results:
            print("  (no matches above threshold)")
        else:
            for r in results:
                print(f"  Game {r['GameNumber']} | FaceID {r['GameFaceID']:>5} | "
                      f"cosine {r['Cosine']:.4f} | euclidean {r['Euclidean']:.4f} | "
                      f"{r['GamePhoto']} | det_conf {r['DetConf']:.3f}")

    # Summary stats
    print("\n" + "=" * 90)
    all_max_sims = []
    for s_id, s_vec in studio_vecs.items():
        sims = game_matrix @ s_vec
        all_max_sims.append(float(sims.max()))
    all_max_sims = np.array(all_max_sims)
    print(f"\nSummary (best match per studio face):")
    print(f"  Count:   {len(all_max_sims)}")
    print(f"  Min:     cosine {all_max_sims.min():.4f}  |  euclidean {np.sqrt(2 - 2*all_max_sims.min()):.4f}")
    print(f"  Max:     cosine {all_max_sims.max():.4f}  |  euclidean {np.sqrt(2 - 2*all_max_sims.max()):.4f}")
    print(f"  Mean:    cosine {all_max_sims.mean():.4f}  |  euclidean {np.sqrt(2 - 2*all_max_sims.mean()):.4f}")
    print(f"  Median:  cosine {np.median(all_max_sims):.4f}  |  euclidean {np.sqrt(2 - 2*np.median(all_max_sims)):.4f}")


if __name__ == "__main__":
    main()
