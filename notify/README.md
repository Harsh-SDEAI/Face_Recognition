# Start / Heartbeat / Stop Email Notifier

Emails you about the AI Face Matching service during its nightly window:

- **start** (~2:05 AM) - confirms the service actually started, or alerts you it
  didn't.
- **check** (hourly, 3/4/5 AM) - a heartbeat: confirms it's still running, or
  alerts you the moment it's down.
- **stop** (~6:02 AM) - a summary of the night's work.

## No false alarms (important)

`start` and `check` **verify the service is genuinely running** (a `tasklist`
process check) *before* they say so. So you never get a "started" email when the
exe actually failed to launch - instead you get a clear **[ALERT]**. That was the
whole point: the email reflects reality.

| Process state | `start` says | `check` says |
|---|---|---|
| Running | "Started OK" / "Running OK" | (same) |
| **Not** running | **"[ALERT] FAILED to start"** | **"[ALERT] is NOT running"** |
| Can't tell | "status could not be verified" | (same) |

## Why it's a separate program

The service is **force-killed** by Task Scheduler at 6:00 AM, so it cannot run
its own "I stopped" email. This notifier therefore runs as its own tiny process,
triggered by Task Scheduler. It does **not** modify, rebuild, or depend on the
main service exe.

The folder is **self-contained**: it has its own `db.py` and its own `.env`, and
imports nothing from outside this folder. You can copy just the `notify` folder
to the server. DB access is **read-only** (`pyodbc readonly=True` + a write/DDL
keyword guard, SELECTs only). Only the **DPP** database is used.

## Setup

1. Install deps (Python is already on the server):
   ```
   pip install -r notify/requirements-notify.txt
   ```
2. Create a `.env` **in this folder** with the DPP credentials and the email
   settings (this is the only DB it needs):
   ```
   # --- DPP database (read-only) ---
   DB_DRIVER_DPP=ODBC Driver 17 for SQL Server
   DB_SERVER_DPP=your_dpp_server
   DB_NAME_DPP=DREAMSPARKPHOTOS
   DB_USER_DPP=your_readonly_user
   DB_PASSWORD_DPP=your_password

   # --- Email (Gmail) ---
   SMTP_USER=masterlysocial@gmail.com
   SMTP_PASSWORD=your_16char_app_password
   NOTIFY_RECIPIENTS=harsh.n@masterlysolutions.com, teammate@yourcompany.com

   # Optional (Gmail defaults shown - no need to set these):
   # SMTP_HOST=smtp.gmail.com
   # SMTP_PORT=465
   # SMTP_FROM=masterlysocial@gmail.com

   # Optional notifier tuning:
   # SERVICE_PROCESS_NAME=AIPhotoMatch2026.exe   # the exe to look for
   # NOTIFY_LOG_DIR=C:\releasebuilds\AIPhotoMatch\logs  # adds "last log activity"
   # NOTIFY_HEARTBEAT_ONLY_ON_FAILURE=true       # DEFAULT: quiet hourly checks
   #                                             # set to false for hourly "Running OK"
   ```

## Test it by hand

**Start with the SMTP test** - it sends a one-line email and prints exactly what
config it used, so you can confirm email works *before* scheduling anything:

```
cd notify
python notifier.py test
```
If it fails, the most common cause on Office 365 is **Authenticated SMTP being
disabled** for the mailbox - ask IT to enable SMTP AUTH for that account (or use
an app password). It also checks for missing creds, wrong user/password, and a
blocked port 587.

Once `test` works, the rest will too:
```
python notifier.py start    # should say "Started OK" if the service is up now
python notifier.py check    # heartbeat
python notifier.py stop     # night's summary
```

## Schedule it (Task Scheduler GUI)

Create these tasks, same style as the service task
(General: "Run whether user is logged on or not", highest privileges;
Action "Start in" = the folder that has `.env`):

| Task name                | Trigger (Daily)        | Program | Arguments           |
|--------------------------|------------------------|---------|---------------------|
| `AIPhotoMatch-StartMail` | **2:05 AM**            | `python`| `notifier.py start` |
| `AIPhotoMatch-Heartbeat` | **3:00 / 4:00 / 5:00 AM** (see below) | `python`| `notifier.py check` |
| `AIPhotoMatch-StopMail`  | **6:02 AM**            | `python`| `notifier.py stop`  |

### Heartbeat timing - DO NOT let it fire at 6:00

The heartbeat must run only at **3:00, 4:00, 5:00** - never at **6:00**. At
6:00 the service is being force-killed, so a 6:00 check races the shutdown and
sends a FALSE "[ALERT] NOT running" email.

**Recommended (zero ambiguity):** give the Heartbeat task **three separate Daily
triggers** at **3:00**, **4:00**, and **5:00**.

**Alternative (single repeating trigger):** Start **3:00 AM**, tick **Repeat task
every: 1 hour**, **for a duration of: 2 hours**. That fires at 3:00, 4:00, 5:00
and stops - a "3 hours" duration is WRONG because it also fires at 6:00.

Notes:
- **2:05 / 6:02**, not 2:00 / 6:00 - the start check runs a few minutes *after*
  the service launches, and the stop summary runs a few minutes *after* the
  6:00 kill so the database reflects the final state.
- If `python` isn't on PATH for the service account, use the full path to
  `python.exe` in **Program/script** and keep the `notifier.py ...` part in
  **Arguments**.
- The notifier never throws on a mail/DB error - it just logs and exits, so a
  bad night can't cascade.

## Heartbeat noise level

**Default: quiet** (`NOTIFY_HEARTBEAT_ONLY_ON_FAILURE` defaults to `true`). The
hourly checks stay silent while healthy and only email when the service is
**down**. You still always get the 2:05 start confirmation and the 6:02 summary.

If you'd rather get a "Running OK" email every hour as reassurance, set
`NOTIFY_HEARTBEAT_ONLY_ON_FAILURE=false` in `.env`.

## What the emails contain

- **Start:** "Service confirmed running at HH:MM" - or an **[ALERT]** that it
  failed to start.
- **Heartbeat:** "Running OK at HH:MM" - or an **[ALERT]** that it is not
  running.
- **Stop:** games completed, games errored, player matches, distinct players
  matched, average time per game, and how many games are left in the queue
  (those get picked up automatically at 02:00 the next day).
