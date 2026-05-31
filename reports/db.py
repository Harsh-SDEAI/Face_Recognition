"""
Read-only database access for the season reporting tool.

SAFETY: This module is the ONLY place that talks to the databases, and it is
read-only by design at THREE independent levels:

  1. Connections are opened with pyodbc readonly=True (the ODBC driver itself
     refuses any write).
  2. run_query() rejects any SQL containing a write / DDL keyword before it is
     ever sent to the server.
  3. The tool never calls commit() and never builds an INSERT/UPDATE/DELETE.

Credentials are reused from the production settings.py so there is no second
copy of connection details to maintain.
"""

import os
import re
import sys
import warnings

import pandas as pd
import pyodbc

# --- Reuse the production credentials (settings.py lives one level up) -------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings  # noqa: E402

# --- Guard: any query matching this is refused (defense in depth) ------------
# \b word boundaries make sure column names like "CreatedOn"/"UpdatedOn" are
# NOT mistaken for CREATE/UPDATE.
_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|"
    r"GRANT|REVOKE|EXEC|EXECUTE|SP_|XP_)\b|\bSELECT\b[\s\S]*\bINTO\b",
    re.IGNORECASE,
)


def _conn_str(driver, server, database, uid, pwd):
    return (
        f"DRIVER={driver};SERVER={server};DATABASE={database};"
        f"Uid={uid};Pwd={pwd};"
    )


def connect_dpp():
    """AI state DB (DREAMSPARKPHOTOS) - the main source for this report."""
    return pyodbc.connect(
        _conn_str(
            settings.DRIVER_NAME_DPP, settings.SERVER_NAME_DPP,
            settings.DATABASE_NAME_DPP, settings.USER_NAME_DPP,
            settings.PASSWORD_DPP,
        ),
        readonly=True,
    )


def connect_cdp2000():
    """Tournament operations DB (rosters / teams). Read-only."""
    return pyodbc.connect(
        _conn_str(
            settings.DRIVER_NAME_CDP2000, settings.SERVER_NAME_CDP2000,
            settings.DATABASE_NAME_CDP2000, settings.USER_NAME_CDP2000,
            settings.PASSWORD_CDP2000,
        ),
        readonly=True,
    )


def connect_cdpmc():
    """Photo catalog DB (Constellation). Read-only."""
    return pyodbc.connect(
        _conn_str(
            settings.DRIVER_NAME_CDPMC, settings.SERVER_NAME_CDPMC,
            settings.DATABASE_NAME_CDPMC, settings.USER_NAME_CDPMC,
            settings.PASSWORD_CDPMC,
        ),
        readonly=True,
    )


def run_query(conn, query, params=None):
    """
    Execute a SELECT and return a pandas DataFrame.

    Refuses anything that is not a pure read. Uses the pyodbc cursor directly
    (rather than pandas' SQLAlchemy path) so there is no extra dependency and
    no spurious warnings.
    """
    if _WRITE_PATTERN.search(query):
        raise ValueError(
            "Refusing to run a query that contains a write/DDL keyword. "
            "This reporting tool is strictly read-only."
        )

    cur = conn.cursor()
    try:
        cur.execute(query, params or [])
        columns = [col[0] for col in cur.description]
        rows = [tuple(r) for r in cur.fetchall()]
    finally:
        cur.close()

    return pd.DataFrame.from_records(rows, columns=columns)
