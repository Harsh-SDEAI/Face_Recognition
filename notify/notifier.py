"""
Start/stop email notifier for the AI Face Matching service.

This runs as its OWN small process, triggered by Task Scheduler - deliberately
separate from the main service, because the service is force-killed at 06:00 and
therefore cannot run its own shutdown email.

Usage:
    python notifier.py start    # service is starting for the night
    python notifier.py stop     # service has stopped -> include night's summary

Strictly read-only with respect to the databases.
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Load the same .env the service uses (run from the notify/ folder, or rely on
# Task Scheduler's "Start in" pointing at the deployment folder).
load_dotenv()

import mailer  # noqa: E402


def notify_start():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"AI Face Matching - Service Started ({datetime.now():%Y-%m-%d})"
    text = (f"AI Face Matching service has started for the night at {now}.\n\n"
            f"It will process the queue and stop automatically at 06:00.")
    html = (f"<html><body style='font-family:Segoe UI,Arial,sans-serif;color:#222'>"
            f"<h2 style='margin:0 0 4px'>AI Face Matching - Service Started</h2>"
            f"<p>Started at <b>{now}</b>.</p>"
            f"<p style='color:#666'>It will process the queue and stop "
            f"automatically at 06:00.</p></body></html>")
    mailer.send_email(subject, text, html)


def notify_stop():
    # Import here so a start-only run never needs pyodbc/pandas.
    import summary
    try:
        data = summary.build_summary()
        subject, text, html = summary.render(data)
    except Exception as e:  # noqa: BLE001 - still send a "stopped" note
        subject = f"AI Face Matching - Service Stopped ({datetime.now():%Y-%m-%d})"
        text = ("AI Face Matching service has stopped.\n\n"
                f"(Could not build the summary: {type(e).__name__}: {e})")
        html = None
    mailer.send_email(subject, text, html)


def main():
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "").strip()
    if mode == "start":
        notify_start()
    elif mode == "stop":
        notify_stop()
    else:
        print("Usage: python notifier.py [start|stop]", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
