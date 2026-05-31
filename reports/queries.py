"""
All report queries. Every statement is a SELECT; aggregation is done in SQL,
and the few cross-database combinations (roster size, tournament names) are
merged in pandas because the databases may live on different servers.

Each public function returns a pandas DataFrame. Functions are written to be
called individually so the orchestrator can wrap each in try/except and let one
failing section degrade gracefully without killing the whole report.
"""

import pandas as pd

from db import run_query
import settings


# ---------------------------------------------------------------------------
# TOURNAMENT NAME LOOKUP  (OPTIONAL - PLEASE READ)
# ---------------------------------------------------------------------------
# DPP only stores TournamentID. If you know the table + columns that hold the
# human-readable tournament name (likely in CDP2000), put a SELECT here that
# returns exactly two columns: TournamentID, TournamentName.
# Leave as None to fall back to "Tournament <id>".
#
# Example:
#   TOURNAMENT_NAME_SQL = (
#       "SELECT TournamentID, TournamentName FROM CDP2000.WSA.Tournament"
#   )
# ---------------------------------------------------------------------------
TOURNAMENT_NAME_SQL = None


def _to_num(df, cols):
    """Coerce given columns to numeric (pyodbc returns Decimals)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _pct(numerator, denominator):
    """Safe division returning a fraction (0..1) for percent-formatted cells."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return (num / den.replace(0, pd.NA)).fillna(0)


def get_tournament_names(conn_cdp2000):
    """Return {TournamentID: TournamentName} or {} if not configured/available."""
    if not TOURNAMENT_NAME_SQL:
        return {}
    try:
        df = run_query(conn_cdp2000, TOURNAMENT_NAME_SQL)
        df = df.rename(columns={df.columns[0]: "TournamentID",
                                df.columns[1]: "TournamentName"})
        return dict(zip(df["TournamentID"], df["TournamentName"]))
    except Exception:
        return {}


def _apply_names(df, names):
    """Add a TournamentName column from the lookup, falling back to the ID."""
    if "TournamentID" not in df.columns:
        return df
    df = df.copy()
    df.insert(
        df.columns.get_loc("TournamentID") + 1,
        "TournamentName",
        df["TournamentID"].map(lambda x: names.get(x, f"Tournament {x}")),
    )
    return df


# ===========================================================================
#  OVERVIEW (headline KPIs)  ->  Metric / Value table
# ===========================================================================
def overview(conn_dpp):
    rows = []

    games = run_query(conn_dpp, """
        SELECT
            COUNT(DISTINCT GameNumber) AS Games,
            SUM(CASE WHEN Status = 'completed' THEN 1 ELSE 0 END) AS Completed,
            SUM(CASE WHEN Status = 'error' THEN 1 ELSE 0 END) AS Errored
        FROM AITournamentQueue
    """)
    g = games.iloc[0]
    total = float(g["Games"] or 0)
    completed = float(g["Completed"] or 0)
    rows.append(("Total games processed", int(total)))
    rows.append(("Games completed successfully", int(completed)))
    rows.append(("Games ended in error", int(g["Errored"] or 0)))
    if total:
        rows.append(("Success rate",
                     f"{completed / total * 100:.1f}%"))

    photos = run_query(conn_dpp, """
        SELECT
            COUNT(*) AS Photos,
            SUM(CASE WHEN IsMatch = 1 THEN 1 ELSE 0 END) AS PhotosWithMatch,
            SUM(CAST(NuFaceDetected AS BIGINT)) AS Faces
        FROM GamePhotoDetail
    """)
    p = photos.iloc[0]
    rows.append(("Total game photos processed", int(p["Photos"] or 0)))
    rows.append(("Photos with at least one match", int(p["PhotosWithMatch"] or 0)))
    rows.append(("Total faces detected", int(p["Faces"] or 0)))

    res = run_query(conn_dpp, """
        SELECT
            COUNT(*) AS Suggestions,
            COUNT(DISTINCT RosterID) AS PlayersMatched,
            SUM(CASE WHEN Manual = 1 THEN 1 ELSE 0 END) AS ManualResults
        FROM AIResult
    """)
    r = res.iloc[0]
    rows.append(("Total player suggestions", int(r["Suggestions"] or 0)))
    rows.append(("Distinct players matched", int(r["PlayersMatched"] or 0)))

    timing = run_query(conn_dpp, """
        SELECT AVG(mins) AS AvgMinutes, MIN(mins) AS MinMinutes, MAX(mins) AS MaxMinutes
        FROM (
            SELECT DATEDIFF(SECOND, ProcessStartOn, ProcessEndOn) / 60.0 AS mins
            FROM AITournamentQueue
            WHERE Status = 'completed'
              AND ProcessStartOn IS NOT NULL AND ProcessEndOn IS NOT NULL
        ) t
    """)
    t = timing.iloc[0]
    if t["AvgMinutes"] is not None:
        rows.append(("Average time per game (minutes)",
                     round(float(t["AvgMinutes"]), 2)))
        rows.append(("Fastest game (minutes)", round(float(t["MinMinutes"]), 2)))
        rows.append(("Slowest game (minutes)", round(float(t["MaxMinutes"]), 2)))

    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ===========================================================================
#  BY TOURNAMENT
# ===========================================================================
def by_tournament(conn_dpp, names):
    q = run_query(conn_dpp, """
        SELECT TournamentID,
               COUNT(DISTINCT GameNumber) AS Games,
               SUM(CASE WHEN Status = 'completed' THEN 1 ELSE 0 END) AS Completed,
               SUM(CASE WHEN Status = 'error' THEN 1 ELSE 0 END) AS Errored,
               AVG(CASE WHEN Status = 'completed'
                         AND ProcessStartOn IS NOT NULL AND ProcessEndOn IS NOT NULL
                        THEN DATEDIFF(SECOND, ProcessStartOn, ProcessEndOn) / 60.0 END) AS AvgMinutes,
               MAX(CASE WHEN ProcessStartOn IS NOT NULL AND ProcessEndOn IS NOT NULL
                        THEN DATEDIFF(SECOND, ProcessStartOn, ProcessEndOn) / 60.0 END) AS MaxMinutes
        FROM AITournamentQueue
        GROUP BY TournamentID
    """)
    ph = run_query(conn_dpp, """
        SELECT TournamentID,
               COUNT(*) AS Photos,
               SUM(CASE WHEN IsMatch = 1 THEN 1 ELSE 0 END) AS PhotosWithMatch,
               SUM(CAST(NuFaceDetected AS BIGINT)) AS Faces
        FROM GamePhotoDetail
        GROUP BY TournamentID
    """)
    rs = run_query(conn_dpp, """
        SELECT TournamentID,
               COUNT(*) AS Suggestions,
               COUNT(DISTINCT RosterID) AS PlayersMatched,
               SUM(CASE WHEN Manual = 1 THEN 1 ELSE 0 END) AS ManualResults
        FROM AIResult
        GROUP BY TournamentID
    """)

    df = q.merge(ph, on="TournamentID", how="left").merge(rs, on="TournamentID", how="left")
    df = _to_num(df, ["Games", "Completed", "Errored", "AvgMinutes", "MaxMinutes",
                       "Photos", "PhotosWithMatch", "Faces", "Suggestions",
                       "PlayersMatched", "ManualResults"])
    df["MatchRate%"] = _pct(df["PhotosWithMatch"], df["Photos"])
    df["ManualRate%"] = _pct(df["ManualResults"], df["Suggestions"])
    df = _apply_names(df, names)
    return df.sort_values("Games", ascending=False).reset_index(drop=True)


# ===========================================================================
#  BY TEAM  (+ coverage from CDP2000 roster sizes)
# ===========================================================================
def by_team(conn_dpp, conn_cdp2000, names):
    teams = run_query(conn_dpp, """
        SELECT TournamentID, TeamKey,
               MAX(TeamName) AS TeamName,
               MAX(TeamNumber) AS TeamNumber,
               COUNT(*) AS Suggestions,
               COUNT(DISTINCT RosterID) AS PlayersMatched,
               SUM(CASE WHEN Manual = 1 THEN 1 ELSE 0 END) AS ManualResults,
               SUM(CASE WHEN IsGroup = 1 THEN 1 ELSE 0 END) AS GroupMatches
        FROM AIResult
        GROUP BY TournamentID, TeamKey
    """)

    # roster sizes from CDP2000 (separate server -> merge in pandas)
    try:
        roster = run_query(conn_cdp2000, """
            SELECT TeamKey, COUNT(DISTINCT RosterID) AS RosterSize
            FROM Roster
            GROUP BY TeamKey
        """)
        teams = teams.merge(roster, on="TeamKey", how="left")
    except Exception:
        teams["RosterSize"] = pd.NA  # coverage will simply be blank

    teams = _to_num(teams, ["Suggestions", "PlayersMatched", "ManualResults",
                            "GroupMatches", "RosterSize"])
    teams["Coverage%"] = _pct(teams["PlayersMatched"], teams["RosterSize"])
    teams = _apply_names(teams, names)
    return teams.sort_values(["TournamentID", "PlayersMatched"],
                             ascending=[True, False]).reset_index(drop=True)


# ===========================================================================
#  BY GAME  (full game-level table)
# ===========================================================================
def by_game(conn_dpp, names):
    gpd = run_query(conn_dpp, """
        SELECT GameNumber, TournamentID,
               COUNT(*) AS Photos,
               SUM(CASE WHEN IsMatch = 1 THEN 1 ELSE 0 END) AS PhotosWithMatch,
               SUM(CAST(NuFaceDetected AS BIGINT)) AS Faces,
               SUM(CASE WHEN IsGroupPhoto = 1 THEN 1 ELSE 0 END) AS GroupPhotos,
               SUM(CASE WHEN U = 1 THEN 1 ELSE 0 END) AS UmpirePhotos
        FROM GamePhotoDetail
        GROUP BY GameNumber, TournamentID
    """)
    res = run_query(conn_dpp, """
        SELECT GameNumber,
               COUNT(*) AS Suggestions,
               COUNT(DISTINCT RosterID) AS PlayersMatched
        FROM AIResult
        GROUP BY GameNumber
    """)
    q = run_query(conn_dpp, """
        SELECT GameNumber, Status, RetryCount,
               CASE WHEN ProcessStartOn IS NOT NULL AND ProcessEndOn IS NOT NULL
                    THEN DATEDIFF(SECOND, ProcessStartOn, ProcessEndOn) / 60.0 END AS ProcessMinutes
        FROM AITournamentQueue
    """)

    df = gpd.merge(res, on="GameNumber", how="left").merge(q, on="GameNumber", how="left")
    df = _to_num(df, ["Photos", "PhotosWithMatch", "Faces", "GroupPhotos",
                      "UmpirePhotos", "Suggestions", "PlayersMatched",
                      "RetryCount", "ProcessMinutes"])
    df = _apply_names(df, names)
    return df.sort_values(["TournamentID", "GameNumber"]).reset_index(drop=True)


# ===========================================================================
#  INTERNAL-ONLY SECTIONS
# ===========================================================================
def performance(conn_dpp):
    """Latency distribution computed in pandas from per-game minutes."""
    mins = run_query(conn_dpp, """
        SELECT DATEDIFF(SECOND, ProcessStartOn, ProcessEndOn) / 60.0 AS mins
        FROM AITournamentQueue
        WHERE Status = 'completed'
          AND ProcessStartOn IS NOT NULL AND ProcessEndOn IS NOT NULL
    """)
    s = pd.to_numeric(mins["mins"], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame({"Metric": ["No completed games found"], "Value": [0]})
    rows = [
        ("CompletedGames", int(s.count())),
        ("AvgMinutes", round(float(s.mean()), 2)),
        ("MinMinutes", round(float(s.min()), 2)),
        ("Median", round(float(s.median()), 2)),
        ("p90Minutes", round(float(s.quantile(0.90)), 2)),
        ("p95Minutes", round(float(s.quantile(0.95)), 2)),
        ("p99Minutes", round(float(s.quantile(0.99)), 2)),
        ("MaxMinutes", round(float(s.max()), 2)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def reliability(conn_dpp):
    status = run_query(conn_dpp, """
        SELECT Status, COUNT(*) AS Games
        FROM AITournamentQueue
        GROUP BY Status
    """)
    retries = run_query(conn_dpp, """
        SELECT RetryCount, COUNT(*) AS Games
        FROM AITournamentQueue
        GROUP BY RetryCount
    """)
    retries = _to_num(retries, ["RetryCount", "Games"]).sort_values("RetryCount")

    failed = run_query(conn_dpp, """
        SELECT COUNT(*) AS FailedPermanently
        FROM AITournamentQueue
        WHERE Status = 'error' AND RetryCount >= ?
    """, [settings.RETRY_COUNT])

    # stack the three little tables into one labelled sheet
    out = []
    for _, row in status.iterrows():
        out.append(("Games with status: " + str(row["Status"]), int(row["Games"])))
    for _, row in retries.iterrows():
        out.append(("Games with RetryCount = " + str(int(row["RetryCount"])),
                    int(row["Games"])))
    out.append(("Games failed permanently (used all retries)",
                int(failed.iloc[0]["FailedPermanently"] or 0)))
    return pd.DataFrame(out, columns=["Metric", "Value"])


def data_quality(conn_dpp):
    out = []
    try:
        miss = run_query(conn_dpp, "SELECT COUNT(*) AS c FROM MissingImagePath")
        out.append(("Missing image paths", int(miss.iloc[0]["c"] or 0)))
    except Exception:
        pass
    try:
        nullemb = run_query(conn_dpp, """
            SELECT COUNT(*) AS c FROM PlayerPhotoEmbedding WHERE SFaceEmbeddings IS NULL
        """)
        out.append(("Players with no usable studio face", int(nullemb.iloc[0]["c"] or 0)))
    except Exception:
        pass
    try:
        zero = run_query(conn_dpp, """
            SELECT COUNT(*) AS c FROM GamePhotoDetail WHERE NuFaceDetected = 0
        """)
        out.append(("Game photos with zero faces detected", int(zero.iloc[0]["c"] or 0)))
    except Exception:
        pass
    if not out:
        out.append(("No data-quality metrics available", 0))
    return pd.DataFrame(out, columns=["Metric", "Value"])


def umpire_group(conn_dpp):
    df = run_query(conn_dpp, """
        SELECT
            SUM(CASE WHEN U = 1 THEN 1 ELSE 0 END) AS UmpirePhotos,
            SUM(CASE WHEN IsGroupPhoto = 1 THEN 1 ELSE 0 END) AS GroupPhotos,
            SUM(CASE WHEN R = 1 THEN 1 ELSE 0 END) AS HomeTeamPhotos,
            SUM(CASE WHEN B = 1 THEN 1 ELSE 0 END) AS VisitorTeamPhotos
        FROM GamePhotoDetail
    """)
    row = df.iloc[0]
    out = [
        ("UmpirePhotos", int(row["UmpirePhotos"] or 0)),
        ("GroupPhotos", int(row["GroupPhotos"] or 0)),
        ("HomeTeamPhotos", int(row["HomeTeamPhotos"] or 0)),
        ("VisitorTeamPhotos", int(row["VisitorTeamPhotos"] or 0)),
    ]
    return pd.DataFrame(out, columns=["Metric", "Value"])


def throughput(conn_dpp):
    df = run_query(conn_dpp, """
        SELECT CAST(ProcessEndOn AS DATE) AS Day,
               COUNT(DISTINCT GameNumber) AS GamesProcessed
        FROM AITournamentQueue
        WHERE ProcessEndOn IS NOT NULL
        GROUP BY CAST(ProcessEndOn AS DATE)
        ORDER BY CAST(ProcessEndOn AS DATE)
    """)
    return _to_num(df, ["GamesProcessed"])
