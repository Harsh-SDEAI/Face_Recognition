"""Single pyodbc connection helper.  Reads credentials from config (which reads .env)."""
import pyodbc

import config


def _conn_str() -> str:
    # SQL Server auth (matches production pattern)
    return (
        f"DRIVER={{{config.DB_DRIVER}}};"
        f"SERVER={config.DB_SERVER};"
        f"DATABASE={config.DB_NAME};"
        f"UID={config.DB_USER};"
        f"PWD={config.DB_PASSWORD};"
    )


def connect() -> pyodbc.Connection:
    return pyodbc.connect(_conn_str())
