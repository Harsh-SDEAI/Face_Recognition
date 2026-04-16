"""Streamlit UI for manual face-match review across 5 models.

Same as streamlit_app.py, plus a "Compare" button on each match card that
pops up a modal dialog showing studio face (crop + full) on the left and the
selected game face (crop + full) on the right, side-by-side for easy review.

Run:
    streamlit run new_streamlit_app.py
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
    """Return latest Y/N judgment per (game_face_id, model_name) for this studio face."""
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT GameFaceID, ModelName, Judgment "
        "FROM EvalManualJudgment "
        "WHERE StudioFaceID = ? "
        "ORDER BY JudgedAt ASC",
        studio_face_id,
    )
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
    return float(np.dot(a, b))


# ---------- Similarity precompute (Option 3) ----------
def precompute_game_similarity_data(game_df: pd.DataFrame, embeddings: dict) -> dict:
    """Stack all game-face embeddings into a (N, D) matrix per model so that
    cosine similarity against any studio face becomes a single matrix-vector
    product ``G @ s`` instead of a Python loop that re-runs every rerun.

    Returned structure::

        {
            "game_matrices":         {model_name: np.ndarray (N_m, D) float32},
            "game_face_ids_by_model":{model_name: np.ndarray (N_m,)    int64},
            "game_meta":             {face_id: {ImagePath, BoxX1, ...}},
        }

    ``N_m`` may differ per model because a face is skipped if it has no
    embedding for that model.  Embeddings are already L2-normalized in the DB
    so the dot product equals the cosine similarity.
    """
    game_meta = {
        int(row.FaceID): {
            "ImagePath": row.ImagePath,
            "BoxX1": int(row.BoxX1), "BoxY1": int(row.BoxY1),
            "BoxX2": int(row.BoxX2), "BoxY2": int(row.BoxY2),
            "FaceCropPath": row.FaceCropPath,
        }
        for row in game_df.itertuples()
    }

    game_matrices: dict[str, np.ndarray] = {}
    game_face_ids_by_model: dict[str, np.ndarray] = {}
    for m in MODELS:
        model_embeds = embeddings.get(m, {})
        fids: list[int] = []
        vecs: list[np.ndarray] = []
        for row in game_df.itertuples():
            fid = int(row.FaceID)
            ge = model_embeds.get(fid)
            if ge is None:
                continue
            fids.append(fid)
            vecs.append(ge)
        if vecs:
            game_matrices[m] = np.vstack(vecs).astype(np.float32)
            game_face_ids_by_model[m] = np.array(fids, dtype=np.int64)
        else:
            # Preserve shape so downstream code can still call `G @ s`.
            game_matrices[m] = np.zeros((0, 512), dtype=np.float32)
            game_face_ids_by_model[m] = np.array([], dtype=np.int64)

    return {
        "game_matrices": game_matrices,
        "game_face_ids_by_model": game_face_ids_by_model,
        "game_meta": game_meta,
    }


# ---------- Compare dialog ----------
@st.dialog("Side-by-side comparison", width="large")
def compare_dialog(studio_info: dict, game_info: dict,
                   prior_verdict: str | None, threshold: float):
    """Modal: studio (left) vs game (right). Each side shows aligned crop on top,
    full photo with red box below.  Includes badge + ✓/✗ buttons so the reviewer
    can label without closing the dialog."""
    st.caption(
        f"Studio FaceID {studio_info['face_id']} (team {studio_info['team_key']})  "
        f"vs  Game FaceID {game_info['face_id']}  |  "
        f"model {game_info['model_name']}  |  sim {game_info['similarity']:.3f}  |  "
        f"threshold {threshold:.2f}"
    )

    # Prior-judgment badge at the top so it's visible alongside the photos.
    if prior_verdict == "Y":
        st.success("\u2713 already marked correct (click a button below to overwrite)")
    elif prior_verdict == "N":
        st.error("\u2717 already marked wrong (click a button below to overwrite)")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Studio")
        crop = studio_info.get("crop_path")
        if crop and Path(crop).exists():
            st.image(crop, caption="Aligned crop", width=220)
        try:
            full = draw_box(studio_info["image_path"],
                            studio_info["x1"], studio_info["y1"],
                            studio_info["x2"], studio_info["y2"])
            st.image(full, caption=Path(studio_info["image_path"]).name,
                     use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"cannot render studio full photo: {exc}")

    with right:
        st.markdown("#### Game")
        crop = game_info.get("crop_path")
        if crop and Path(crop).exists():
            st.image(crop, caption="Aligned crop", width=220)
        try:
            full = draw_box(game_info["image_path"],
                            game_info["x1"], game_info["y1"],
                            game_info["x2"], game_info["y2"])
            st.image(full, caption=Path(game_info["image_path"]).name,
                     use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"cannot render game full photo: {exc}")

    st.markdown("---")
    b_y, b_n, b_close = st.columns(3)
    with b_y:
        if st.button("\u2713 correct", key="compare_y", use_container_width=True):
            insert_judgment(studio_info["face_id"], game_info["face_id"],
                            game_info["model_name"], "Y",
                            game_info["similarity"], threshold)
            st.rerun()
    with b_n:
        if st.button("\u2717 wrong", key="compare_n", use_container_width=True):
            insert_judgment(studio_info["face_id"], game_info["face_id"],
                            game_info["model_name"], "N",
                            game_info["similarity"], threshold)
            st.rerun()
    with b_close:
        if st.button("Close", key="compare_close", use_container_width=True):
            st.rerun()


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

    # Pack studio info once for reuse in the compare dialog.
    studio_info = {
        "face_id": int(studio_row.FaceID),
        "team_key": int(studio_row.TeamKey),
        "crop_path": studio_row.FaceCropPath,
        "image_path": studio_row.ImagePath,
        "x1": int(studio_row.BoxX1), "y1": int(studio_row.BoxY1),
        "x2": int(studio_row.BoxX2), "y2": int(studio_row.BoxY2),
    }

    # --- Threshold sliders per model ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Per-model threshold (cosine)")
    thresholds = {
        m: st.sidebar.slider(m, 0.0, 1.0, config.DEFAULT_COSINE_THRESHOLD, 0.01, key=f"thr_{m}")
        for m in MODELS
    }
    top_k = st.sidebar.number_input("Max matches to show per model", 1, 50, 10)

    # --- Precompute per-game similarity matrices (Option 3) ---
    # Key on (game_number, min_conf) because those two inputs define the face
    # set and embedding layout.  Threshold + top_k only filter the result, so
    # they do NOT invalidate the cache.  Stored in st.session_state so the
    # matrix survives reruns until the user picks a different game/confidence.
    sim_cache_key = f"simcache_{game_number}_{min_conf:.4f}"
    if sim_cache_key not in st.session_state:
        st.session_state[sim_cache_key] = precompute_game_similarity_data(
            game_df, embeddings
        )
    sim_cache = st.session_state[sim_cache_key]

    empty_cols = ["GameFaceID", "Similarity", "ImagePath",
                  "BoxX1", "BoxY1", "BoxX2", "BoxY2", "FaceCropPath"]

    # --- Compute matches per model (vectorized) ---
    matches_by_model: dict[str, pd.DataFrame] = {}
    for m in MODELS:
        studio_embed = embeddings.get(m, {}).get(studio_face_id)
        G = sim_cache["game_matrices"][m]
        fids = sim_cache["game_face_ids_by_model"][m]
        if studio_embed is None or G.shape[0] == 0:
            matches_by_model[m] = pd.DataFrame(columns=empty_cols)
            continue

        # Embeddings are L2-normalized, so dot product == cosine similarity.
        sims = G @ studio_embed.astype(np.float32)        # shape (N_game,)
        mask = sims >= thresholds[m]
        if not mask.any():
            matches_by_model[m] = pd.DataFrame(columns=empty_cols)
            continue

        passing_idx = np.where(mask)[0]
        # argsort descending by similarity, then take top_k
        order = passing_idx[np.argsort(-sims[passing_idx])][:int(top_k)]

        meta = sim_cache["game_meta"]
        rows = []
        for idx in order:
            fid = int(fids[idx])
            m_row = meta[fid]
            rows.append({
                "GameFaceID": fid,
                "Similarity": float(sims[idx]),
                "ImagePath": m_row["ImagePath"],
                "BoxX1": m_row["BoxX1"], "BoxY1": m_row["BoxY1"],
                "BoxX2": m_row["BoxX2"], "BoxY2": m_row["BoxY2"],
                "FaceCropPath": m_row["FaceCropPath"],
            })
        matches_by_model[m] = pd.DataFrame(rows, columns=empty_cols)

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

    # --- 5-column grid (text-only; images deferred to Compare dialog) ---
    # Rationale: every button click triggers a full rerun.  Rendering ~50 game
    # thumbnails (up to 5 models x top_k) via draw_box() + JPEG re-encode on
    # every rerun is the dominant cost once similarities are cached.  Keep the
    # grid cheap — just FaceID, similarity, prior-judgment badge, and a
    # Compare button.  Full-size photos load only when the reviewer asks.
    st.subheader("Model matches")
    prior_judgments = load_judgments_for_studio(studio_face_id)
    cols = st.columns(5)
    for col, model_name in zip(cols, MODELS):
        with col:
            st.markdown(f"### {model_name}")
            st.caption(f"threshold {thresholds[model_name]:.2f}")
            df = matches_by_model[model_name]
            if df.empty:
                st.info("No matches above threshold.")
                continue
            for _, row in df.iterrows():
                st.markdown(
                    f"**GameFaceID {int(row.GameFaceID)}** — sim {row.Similarity:.3f}"
                )
                prior = prior_judgments.get((int(row.GameFaceID), model_name))
                if prior == "Y":
                    st.success("\u2713 already marked correct")
                elif prior == "N":
                    st.error("\u2717 already marked wrong")

                key_compare = f"{model_name}_{studio_face_id}_{int(row.GameFaceID)}_cmp"
                if st.button("\U0001F50D Compare", key=key_compare,
                             use_container_width=True):
                    game_info = {
                        "face_id": int(row.GameFaceID),
                        "model_name": model_name,
                        "similarity": float(row.Similarity),
                        "crop_path": row.FaceCropPath,
                        "image_path": row.ImagePath,
                        "x1": int(row.BoxX1), "y1": int(row.BoxY1),
                        "x2": int(row.BoxX2), "y2": int(row.BoxY2),
                    }
                    compare_dialog(studio_info, game_info,
                                   prior_verdict=prior,
                                   threshold=thresholds[model_name])
                st.markdown("---")

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
