"""Streamlit UI for manual face-match review across 5 models.

Layout:
    - Sidebar: game + studio-face selector, view mode, detection-confidence filter
    - Main: selected studio face on top, then 5-column grid (one per model)
    - Each column: threshold slider, sorted matches, bounding box overlay,
      check/cross buttons that insert into EvalManualJudgment.
    - Footer: export CSV / Markdown report.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import config
from utils.db import connect
from utils.report import export_csv, export_markdown

st.set_page_config(page_title="Face Recognition Eval", layout="wide")

MODELS = ["facenet", "arcface", "adaface", "magface", "lvface"]


# ---------- Data access (cached) ----------
@st.cache_data(ttl=60)
def load_games():
    conn = connect()
    df = pd.read_sql("SELECT GameNumber, TeamKey1, TeamKey2 FROM EvalGames ORDER BY GameNumber", conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_studio_faces_for_game(game_number: int, min_conf: float) -> pd.DataFrame:
    """Studio faces belonging to either team of the selected game."""
    conn = connect()
    df = pd.read_sql(
        """
        SELECT f.FaceID, f.Confidence, f.FaceCropPath, f.BoxX1, f.BoxY1, f.BoxX2, f.BoxY2,
               sp.StudioPhotoID, sp.TeamKey, sp.ImagePath
        FROM EvalFaceDetection f
        JOIN EvalStudioPhoto sp ON sp.StudioPhotoID = f.SourceID
        WHERE f.SourceType = 'S' AND f.Confidence >= ?
          AND sp.TeamKey IN (
              SELECT TeamKey1 FROM EvalGames WHERE GameNumber = ?
              UNION
              SELECT TeamKey2 FROM EvalGames WHERE GameNumber = ?
          )
        ORDER BY sp.TeamKey, sp.ImagePath, f.FaceID
        """,
        conn, params=[min_conf, game_number, game_number],
    )
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_game_faces(game_number: int, min_conf: float) -> pd.DataFrame:
    conn = connect()
    df = pd.read_sql(
        """
        SELECT f.FaceID, f.Confidence, f.FaceCropPath, f.BoxX1, f.BoxY1, f.BoxX2, f.BoxY2,
               gp.GamePhotoID, gp.ImagePath
        FROM EvalFaceDetection f
        JOIN EvalGamePhoto gp ON gp.GamePhotoID = f.SourceID
        WHERE f.SourceType = 'G' AND f.Confidence >= ? AND gp.GameNumber = ?
        ORDER BY gp.ImagePath, f.FaceID
        """,
        conn, params=[min_conf, game_number],
    )
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_embeddings_for_faces(face_ids: tuple[int, ...]) -> dict:
    """Returns: {model_name: {face_id: np.ndarray(512,)}}.

    Also returns quality norms keyed under '__qnorm__'.
    """
    if not face_ids:
        return {m: {} for m in MODELS}
    conn = connect()
    cur = conn.cursor()
    result = {m: {} for m in MODELS}
    qnorm = {}
    # SQL Server caps a single query at 2100 parameters; chunk the IN (...) list.
    CHUNK = 1000
    for start in range(0, len(face_ids), CHUNK):
        chunk = face_ids[start:start + CHUNK]
        placeholders = ",".join(["?"] * len(chunk))
        q = (
            "SELECT FaceID, ModelName, Embedding, QualityNorm FROM EvalEmbedding "
            f"WHERE FaceID IN ({placeholders})"
        )
        cur.execute(q, *chunk)
        for face_id, model_name, blob, qn in cur.fetchall():
            if blob is None:
                continue
            arr = np.frombuffer(bytes(blob), dtype=np.float32)
            result.setdefault(model_name, {})[int(face_id)] = arr
            if qn is not None:
                qnorm[int(face_id)] = float(qn)
    cur.close()
    conn.close()
    result["__qnorm__"] = qnorm
    return result


def insert_judgment(studio_face_id: int, game_face_id: int, model_name: str,
                    judgment: str, similarity: float, threshold: float):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO EvalManualJudgment (StudioFaceID, GameFaceID, ModelName, "
        "Judgment, Similarity, Threshold) VALUES (?, ?, ?, ?, ?, ?)",
        studio_face_id, game_face_id, model_name, judgment, similarity, threshold,
    )
    conn.commit()
    cur.close()
    conn.close()


def load_judgments_for_studio(studio_face_id: int) -> dict:
    """Return latest Y/N judgment per (game_face_id, model_name) for this studio face.

    Latest wins when multiple rows exist for the same triplet - allows users to
    overwrite prior decisions simply by clicking again.  Not cached because
    judgments change as the user clicks, and the query is cheap (indexed).
    """
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT GameFaceID, ModelName, Judgment "
        "FROM EvalManualJudgment "
        "WHERE StudioFaceID = ? "
        "ORDER BY JudgedAt ASC",
        studio_face_id,
    )
    # Iterating in ascending time means later rows overwrite earlier -> latest wins.
    result = {(int(g), m): j for g, m, j in cur.fetchall()}
    cur.close()
    conn.close()
    return result


# ---------- Image helpers ----------
def draw_box(image_path: str, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=6)
    return img


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Both assumed L2-normalized; dot product is cosine similarity.
    return float(np.dot(a, b))


# ---------- UI ----------
def main():
    st.title("Face Recognition Model Evaluation")

    games = load_games()
    if games.empty:
        st.error("EvalGames is empty.  Insert 5 rows and run generate_embeddings.py first.")
        return

    # --- Sidebar ---
    st.sidebar.header("Controls")
    game_number = int(st.sidebar.selectbox(
        "Game", games["GameNumber"], format_func=lambda g: f"Game {g}",
    ))

    min_conf = st.sidebar.slider("Min detection confidence", 0.0, 1.0, 0.85, 0.01)

    view_mode = st.sidebar.radio("View mode", ["Per studio face", "Disagreements only"])

    studio_df = load_studio_faces_for_game(game_number, min_conf)
    game_df = load_game_faces(game_number, min_conf)
    if studio_df.empty or game_df.empty:
        st.warning(f"No studio or game faces for game {game_number} at conf >= {min_conf}.")
        return

    all_face_ids = tuple(int(x) for x in pd.concat([studio_df["FaceID"], game_df["FaceID"]]).unique())
    embeddings = load_embeddings_for_faces(all_face_ids)

    # --- Studio face selector ---
    labels = [
        f"Team {row.TeamKey} | FaceID {row.FaceID} | conf {row.Confidence:.2f}"
        for row in studio_df.itertuples()
    ]
    label_to_faceid = {lbl: int(row.FaceID) for lbl, row in zip(labels, studio_df.itertuples())}
    chosen = st.sidebar.selectbox("Studio face", labels)
    studio_face_id = label_to_faceid[chosen]
    studio_row = studio_df[studio_df["FaceID"] == studio_face_id].iloc[0]

    # --- Top: selected studio face ---
    st.subheader(f"Studio face #{studio_face_id} (team {studio_row.TeamKey})")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        crop = studio_row.FaceCropPath
        if crop and Path(crop).exists():
            st.image(crop, caption="Aligned crop", width=180)
    with col_s2:
        full = draw_box(studio_row.ImagePath, studio_row.BoxX1, studio_row.BoxY1,
                        studio_row.BoxX2, studio_row.BoxY2)
        st.image(full, caption=Path(studio_row.ImagePath).name, use_container_width=True)

    # --- Threshold sliders per model ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Per-model threshold (cosine)")
    thresholds = {
        m: st.sidebar.slider(m, 0.0, 1.0, config.DEFAULT_COSINE_THRESHOLD, 0.01, key=f"thr_{m}")
        for m in MODELS
    }
    top_k = st.sidebar.number_input("Max matches to show per model", 1, 50, 8)

    # --- Compute matches per model ---
    matches_by_model: dict[str, pd.DataFrame] = {}
    for m in MODELS:
        studio_embed = embeddings.get(m, {}).get(studio_face_id)
        if studio_embed is None:
            matches_by_model[m] = pd.DataFrame()
            continue
        rows = []
        for _, gr in game_df.iterrows():
            ge = embeddings.get(m, {}).get(int(gr.FaceID))
            if ge is None:
                continue
            sim = cosine_sim(studio_embed, ge)
            if sim >= thresholds[m]:
                rows.append({
                    "GameFaceID": int(gr.FaceID),
                    "Similarity": sim,
                    "ImagePath": gr.ImagePath,
                    "BoxX1": int(gr.BoxX1), "BoxY1": int(gr.BoxY1),
                    "BoxX2": int(gr.BoxX2), "BoxY2": int(gr.BoxY2),
                    "FaceCropPath": gr.FaceCropPath,
                })
        if rows:
            df = pd.DataFrame(rows).sort_values("Similarity", ascending=False).head(int(top_k))
        else:
            df = pd.DataFrame(columns=["GameFaceID", "Similarity", "ImagePath",
                                       "BoxX1", "BoxY1", "BoxX2", "BoxY2",
                                       "FaceCropPath"])
        matches_by_model[m] = df

    # --- Disagreement filter ---
    if view_mode == "Disagreements only":
        sets = [set(df["GameFaceID"]) if not df.empty else set() for df in matches_by_model.values()]
        if sets:
            intersection = set.intersection(*sets) if all(sets) else set()
            union = set.union(*sets)
            disagreed = union - intersection
            for m, df in matches_by_model.items():
                if df.empty:
                    continue
                matches_by_model[m] = df[df["GameFaceID"].isin(disagreed)]

    # --- 5-column grid ---
    st.subheader("Model matches")
    # Load prior judgments for this studio face so previously-marked pairs show
    # a badge.  Re-reads from DB on every rerun, which naturally picks up new
    # clicks (Streamlit reruns main() after each button press).
    prior_judgments = load_judgments_for_studio(studio_face_id)
    cols = st.columns(5)
    qnorms = embeddings.get("__qnorm__", {})
    for col, model_name in zip(cols, MODELS):
        with col:
            st.markdown(f"### {model_name}")
            st.caption(f"threshold {thresholds[model_name]:.2f}")
            df = matches_by_model[model_name]
            if df.empty:
                st.info("No matches above threshold.")
                continue
            for _, row in df.iterrows():
                caption = f"sim {row.Similarity:.3f}"
                if model_name == "adaface":
                    qn = qnorms.get(int(row.GameFaceID))
                    if qn is not None:
                        caption += f" | q {qn:.1f}"
                try:
                    game_img = draw_box(row.ImagePath, row.BoxX1, row.BoxY1,
                                        row.BoxX2, row.BoxY2)
                    st.image(game_img, caption=caption, use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"cannot render: {exc}")
                prior = prior_judgments.get((int(row.GameFaceID), model_name))
                if prior == "Y":
                    st.success("\u2713 already marked correct (click to overwrite)")
                elif prior == "N":
                    st.error("\u2717 already marked wrong (click to overwrite)")
                b1, b2 = st.columns(2)
                key_y = f"{model_name}_{studio_face_id}_{int(row.GameFaceID)}_y"
                key_n = f"{model_name}_{studio_face_id}_{int(row.GameFaceID)}_n"
                with b1:
                    if st.button("\u2713 correct", key=key_y):
                        insert_judgment(studio_face_id, int(row.GameFaceID),
                                        model_name, "Y", float(row.Similarity),
                                        thresholds[model_name])
                        st.success("recorded")
                with b2:
                    if st.button("\u2717 wrong", key=key_n):
                        insert_judgment(studio_face_id, int(row.GameFaceID),
                                        model_name, "N", float(row.Similarity),
                                        thresholds[model_name])
                        st.warning("recorded")

    # --- Footer: export ---
    st.markdown("---")
    st.subheader("Report export")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Export CSV"):
            out = export_csv(Path("eval_report.csv"))
            st.success(f"Written: {out.resolve()}")
    with c2:
        if st.button("Export Markdown"):
            out = export_markdown(Path("eval_report.md"))
            st.success(f"Written: {out.resolve()}")


if __name__ == "__main__":
    main()
