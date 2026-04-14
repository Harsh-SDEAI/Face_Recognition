"""Run db_setup.sql against the evaluation DB.  One-time bootstrap."""
from pathlib import Path

from utils.db import connect

SQL_FILE = Path(__file__).resolve().parent / "db_setup.sql"


def split_batches(sql_text: str):
    """pyodbc cannot run multiple statements at once when they include DDL
    like IF OBJECT_ID ... DROP TABLE; we split on 'GO'-style boundaries.
    The provided file uses semicolons per statement - we split on those."""
    return [s.strip() for s in sql_text.split(";") if s.strip()]


def main():
    sql = SQL_FILE.read_text(encoding="utf-8")
    statements = split_batches(sql)
    conn = connect()
    cur = conn.cursor()
    for stmt in statements:
        try:
            cur.execute(stmt)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] statement failed: {stmt[:80]}... -> {exc}")
    conn.commit()
    cur.close()
    conn.close()
    print("Schema created.  Now INSERT 5 rows into EvalGames manually.")


if __name__ == "__main__":
    main()
