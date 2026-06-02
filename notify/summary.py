"""
Builds the "what happened tonight" summary for the stop email.

Self-contained and strictly READ-ONLY: it uses this folder's own db.py
(pyodbc readonly=True + a write/DDL keyword guard), runs only SELECTs against
the DPP database, and never commits anything.

The "tonight" window is [today 02:00, now]. Because the stop notifier runs at
~06:02, this captures exactly the run that just ended.
"""

from datetime import datetime, time


def _window_start():
    """Today at 02:00 local time (when the service starts)."""
    now = datetime.now()
    return datetime.combine(now.date(), time(2, 0, 0))


def _int(row, col, default=0):
    try:
        v = row.get(col)
        return int(v) if v is not None else default
    except Exception:
        return default


def _first(rows):
    return rows[0] if rows else {}


def build_summary():
    """
    Returns a dict of the night's numbers. Any single query that fails is
    swallowed so a schema mismatch can't stop the email from going out.
    """
    import db  # local, self-contained read-only layer

    start = _window_start()
    data = {
        "window_start": start,
        "generated_at": datetime.now(),
    }

    conn = db.connect_dpp()
    try:
        # Games finished tonight, split by outcome.
        try:
            row = _first(db.run_query(conn, """
                SELECT
                    SUM(CASE WHEN Status = 'completed' THEN 1 ELSE 0 END) AS Completed,
                    SUM(CASE WHEN Status = 'error' THEN 1 ELSE 0 END) AS Errored,
                    AVG(CASE WHEN Status = 'completed'
                              AND ProcessStartOn IS NOT NULL AND ProcessEndOn IS NOT NULL
                             THEN DATEDIFF(SECOND, ProcessStartOn, ProcessEndOn) / 60.0 END)
                        AS AvgMinutes
                FROM AITournamentQueue
                WHERE ProcessEndOn >= ?
            """, [start]))
            data["games_completed"] = _int(row, "Completed")
            data["games_errored"] = _int(row, "Errored")
            avg = row.get("AvgMinutes")
            data["avg_minutes"] = round(float(avg), 2) if avg is not None else None
        except Exception as e:  # noqa: BLE001
            data["games_query_error"] = f"{type(e).__name__}: {e}"

        # Matches / player suggestions created tonight.
        try:
            row = _first(db.run_query(conn, """
                SELECT COUNT(*) AS Suggestions,
                       COUNT(DISTINCT RosterID) AS PlayersMatched
                FROM AIResult
                WHERE CreatedOn >= ?
            """, [start]))
            data["suggestions"] = _int(row, "Suggestions")
            data["players_matched"] = _int(row, "PlayersMatched")
        except Exception as e:  # noqa: BLE001
            data["result_query_error"] = f"{type(e).__name__}: {e}"

        # Whatever is still waiting - picked up at 02:00 tomorrow.
        try:
            row = _first(db.run_query(conn, """
                SELECT COUNT(*) AS Leftover
                FROM AITournamentQueue
                WHERE Status IN ('pending', 'error', 'InProgress')
            """))
            data["leftover_in_queue"] = _int(row, "Leftover")
        except Exception as e:  # noqa: BLE001
            data["queue_query_error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return data


def _fmt(n):
    return f"{n:,}" if isinstance(n, int) else str(n)


def render(data):
    """Return (subject, text_body, html_body) for the stop email."""
    ws = data["window_start"].strftime("%Y-%m-%d %H:%M")
    completed = data.get("games_completed", "n/a")
    errored = data.get("games_errored", "n/a")
    leftover = data.get("leftover_in_queue", "n/a")
    suggestions = data.get("suggestions", "n/a")
    players = data.get("players_matched", "n/a")
    avg = data.get("avg_minutes")
    avg_txt = f"{avg} min" if avg is not None else "n/a"

    subject = (f"AI Face Matching - Service Stopped "
               f"({data['window_start'].strftime('%Y-%m-%d')})")

    lines = [
        "AI Face Matching service has stopped for the night.",
        "",
        f"Run window: {ws} - 06:00",
        "",
        f"  Games completed:                {_fmt(completed)}",
        f"  Games ended in error:           {_fmt(errored)}",
        f"  Player matches (suggestions):   {_fmt(suggestions)}",
        f"  Distinct players matched:       {_fmt(players)}",
        f"  Avg time per completed game:    {avg_txt}",
        f"  Games left in queue:            {_fmt(leftover)}",
        "",
        "Any games left in the queue are processed automatically on the next "
        "run at 02:00.",
    ]
    # Surface any query problems quietly at the bottom (internal email only).
    notes = [v for k, v in data.items() if k.endswith("_error")]
    if notes:
        lines += ["", "Notes:"] + [f"  - {n}" for n in notes]

    text_body = "\n".join(lines)

    html_body = f"""\
<html><body style="font-family:Segoe UI,Arial,sans-serif;color:#222">
  <h2 style="margin:0 0 4px">AI Face Matching - Service Stopped</h2>
  <p style="color:#666;margin:0 0 14px">Run window: {ws} &ndash; 06:00</p>
  <table cellpadding="6" style="border-collapse:collapse">
    <tr><td>Games completed</td><td align="right"><b>{_fmt(completed)}</b></td></tr>
    <tr><td>Games ended in error</td><td align="right"><b>{_fmt(errored)}</b></td></tr>
    <tr><td>Player matches (suggestions)</td><td align="right"><b>{_fmt(suggestions)}</b></td></tr>
    <tr><td>Distinct players matched</td><td align="right"><b>{_fmt(players)}</b></td></tr>
    <tr><td>Avg time per completed game</td><td align="right"><b>{avg_txt}</b></td></tr>
    <tr><td>Games left in queue</td><td align="right"><b>{_fmt(leftover)}</b></td></tr>
  </table>
  <p style="color:#666;margin-top:14px">Any games left in the queue are
  processed automatically on the next run at 02:00.</p>
</body></html>"""

    return subject, text_body, html_body
