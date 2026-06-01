# Start / Stop Email Notifier

Sends an email when the AI Face Matching service **starts** (2:00 AM) and
another, with a summary of the night's work, when it **stops** (6:00 AM).

## Why it's a separate program (important)

The service is **force-killed** by Task Scheduler at 6:00 AM, so it cannot run
its own "I stopped" email. This notifier therefore runs as its own tiny process,
triggered by Task Scheduler at the right times. It does **not** modify, rebuild,
or depend on the main service exe.

It is **read-only** against the databases: it reuses `reports/db.py`
(`pyodbc readonly=True` + a write/DDL keyword guard) and only runs `SELECT`s.

## Setup

1. Install deps (Python is already on the server):
   ```
   pip install -r notify/requirements-notify.txt
   ```
2. Add these keys to the same `.env` the service uses:
   ```
   SMTP_USER=alerts@yourcompany.com
   SMTP_PASSWORD=your_app_or_mailbox_password
   NOTIFY_RECIPIENTS=harsh.n@masterlysolutions.com, teammate@yourcompany.com
   # Optional (Office 365 defaults shown):
   # SMTP_HOST=smtp.office365.com
   # SMTP_PORT=587
   # SMTP_FROM=alerts@yourcompany.com
   ```

## Test it by hand

```
cd notify
python notifier.py start
python notifier.py stop
```
You should receive both emails. The stop email includes the night's numbers.

## Schedule it (Task Scheduler GUI)

Create **two** more tasks, same style as the service task
(General: "Run whether user is logged on or not", highest privileges;
Action "Start in" = the folder that has `.env`):

| Task name                | Trigger (Daily) | Action: Program | Action: Arguments | Start in            |
|--------------------------|-----------------|-----------------|-------------------|---------------------|
| `AIPhotoMatch-StartMail` | **2:00 AM**     | `python`        | `notifier.py start` | the `notify` folder |
| `AIPhotoMatch-StopMail`  | **6:02 AM**     | `python`        | `notifier.py stop`  | the `notify` folder |

Notes:
- **6:02**, not 6:00 - run the stop email a couple of minutes *after* the
  service is killed so the database reflects the final state.
- If `python` isn't on PATH for the service account, use the full path to
  `python.exe` in **Program/script** and keep `notifier.py stop` in
  **Arguments**.
- The notifier never throws on a mail/DB error - it just logs and exits, so a
  bad night can't cascade.

## What the emails contain

- **Start:** "Service started at HH:MM; will stop automatically at 06:00."
- **Stop:** games completed, games errored, player matches, distinct players
  matched, average time per game, and how many games are left in the queue
  (those get picked up automatically at 02:00 the next day).
