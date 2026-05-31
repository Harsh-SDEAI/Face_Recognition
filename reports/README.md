# Season Reports

Two read-only reporting scripts that pull last season's data and generate
Excel workbooks.

- **`client_report.py`** → client-facing workbook (value delivered: photos,
  matches, coverage, turnaround).
- **`internal_report.py`** → everything in the client report **plus**
  performance percentiles, error/retry breakdown, data-quality, and the
  umpire/group photo split.

## Read-only guarantee

This tool never writes to any database. It is read-only at three levels:
1. connections opened with `pyodbc readonly=True`,
2. `db.run_query()` refuses any SQL containing a write/DDL keyword,
3. there are no `INSERT/UPDATE/DELETE` statements and no `commit()` anywhere.

For maximum safety, run it under a SQL login that only has `db_datareader`.

## Setup

```bash
pip install -r reports/requirements-report.txt
```

It reuses the app's `settings.py` + `.env`, so run it from a machine that has
those (e.g. the production/build server).

## Run

```bash
cd reports
python client_report.py            # writes to current folder
python internal_report.py D:\out   # optional output folder
```

Each run produces `CDP_FaceMatching_<Client|Internal>_Report_YYYYMMDD.xlsx`.

## Two things you may need to adjust

1. **Tournament names** — DPP only stores `TournamentID`. To show real names,
   set `TOURNAMENT_NAME_SQL` at the top of `queries.py` to a `SELECT` that
   returns `TournamentID, TournamentName` (likely from CDP2000). Left unset,
   the report falls back to `Tournament <id>`.

2. **Schema differences** — the queries were written from the column names used
   in `AIPhotoMatch.py`. If a column/table differs on the server, only that one
   sheet will show an error note (the rest of the report is unaffected). Send
   that note to the developer and it's a one-line fix.

## Every column is explained

Each data sheet has a hover comment on every column header, and the first
**Legend** tab lists every column with its plain-English meaning. Definitions
live in one place — `column_definitions.py`.
