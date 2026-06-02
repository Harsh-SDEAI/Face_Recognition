"""
Self-contained read-only access to the DPP database for the notifier.

This makes the notify/ folder standalone - it does NOT import anything outside
this folder. Credentials come straight from the environment (.env in this same
folder), and only the DPP database is needed for the stop summary.

READ-ONLY at two levels:
  1. the connection is opened with pyodbc readonly=True, and
  2. run_query() refuses any SQL containing a write / DDL keyword.
There is no commit() and no INSERT/UPDATE/DELETE anywhere.
"""

import os
import re

import pyodbc

# \b word boundaries so column names like CreatedOn / UpdatedOn are NOT mistaken
# for CREATE / UPDATE.
_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|"
    r"GRANT|REVOKE|EXEC|EXECUTE|SP_|XP_)\b|\bSELECT\b[\s\S]*\bINTO\b",
    re.IGNORECASE,
)


def connect_dpp():
    """Open a read-only connection to the DPP database using .env credentials."""
    driver = os.getenv("DB_DRIVER_DPP")
    server = os.getenv("DB_SERVER_DPP")
    database = os.getenv("DB_NAME_DPP")
    uid = os.getenv("DB_USER_DPP")
    pwd = os.getenv("DB_PASSWORD_DPP")
    if not all([driver, server, database, uid, pwd]):
        raise RuntimeError(
            "Missing DPP credentials in .env "
            "(need DB_DRIVER_DPP, DB_SERVER_DPP, DB_NAME_DPP, "
            "DB_USER_DPP, DB_PASSWORD_DPP)."
        )
    conn_str = (
        f"DRIVER={driver};SERVER={server};DATABASE={database};"
        f"Uid={uid};Pwd={pwd};"
    )
    return pyodbc.connect(conn_str, readonly=True)


def run_query(conn, query, params=None):
    """
    Execute a SELECT and return a list of dict rows. Refuses anything that is
    not a pure read. No pandas dependency - the summary only needs a few small
    aggregate rows.
    """
    if _WRITE_PATTERN.search(query):
        raise ValueError(
            "Refusing to run a query that contains a write/DDL keyword. "
            "This notifier is strictly read-only."
        )
    cur = conn.cursor()
    try:
        cur.execute(query, params or [])
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()
