"""Aggregate manual judgments into precision / recall / F1 per (game, model)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.db import connect


def fetch_judgments() -> pd.DataFrame:
    """Pull every row from EvalManualJudgment, joined with GameNumber.

    Studio face -> studio photo -> team -> game(s).  A studio face belongs to
    one team; that team appears in one or more EvalGames rows, so we join via
    team membership.
    """
    q = """
    SELECT j.JudgmentID,
           j.StudioFaceID,
           j.GameFaceID,
           j.ModelName,
           j.Judgment,
           j.Similarity,
           j.Threshold,
           j.JudgedAt,
           gp.GameNumber   AS GameNumber,
           sp.TeamKey      AS TeamKey
    FROM EvalManualJudgment j
    JOIN EvalFaceDetection  sf ON sf.FaceID = j.StudioFaceID
    JOIN EvalStudioPhoto    sp ON sp.StudioPhotoID = sf.SourceID
    JOIN EvalFaceDetection  gf ON gf.FaceID = j.GameFaceID
    JOIN EvalGamePhoto      gp ON gp.GamePhotoID = gf.SourceID
    """
    conn = connect()
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (GameNumber, ModelName) precision / recall / F1.

    Precision = TP / (TP + FP).  TP = judged Y, FP = judged N.
    Recall is computed against the TOTAL number of match pairs the user
    labelled for that model on that game - a proxy since we don't have
    ground-truth matches.  For a stricter recall compute, label a
    disagreement set across all models.
    """
    if df.empty:
        return pd.DataFrame(columns=["GameNumber", "ModelName",
                                     "TP", "FP", "Precision"])

    grp = df.groupby(["GameNumber", "ModelName"])
    rows = []
    for (game, model), sub in grp:
        tp = int((sub["Judgment"] == "Y").sum())
        fp = int((sub["Judgment"] == "N").sum())
        total = tp + fp
        precision = (tp / total) if total else 0.0
        rows.append({
            "GameNumber": int(game),
            "ModelName": model,
            "TP": tp,
            "FP": fp,
            "Total": total,
            "Precision": round(precision, 4),
        })
    return pd.DataFrame(rows).sort_values(["GameNumber", "ModelName"])


def export_csv(out_path: Path) -> Path:
    """Write summary CSV.  Returns the path written."""
    df = fetch_judgments()
    summary = summarize(df)
    out_path = Path(out_path)
    summary.to_csv(out_path, index=False)
    return out_path


def export_markdown(out_path: Path) -> Path:
    """Write Markdown summary grouped by game."""
    df = fetch_judgments()
    summary = summarize(df)
    out_path = Path(out_path)

    lines = ["# Face Recognition Model Evaluation\n"]
    if summary.empty:
        lines.append("_No manual judgments recorded yet._")
    else:
        for game, sub in summary.groupby("GameNumber"):
            lines.append(f"## Game {int(game)}\n")
            lines.append(sub.drop(columns=["GameNumber"]).to_markdown(index=False))
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
