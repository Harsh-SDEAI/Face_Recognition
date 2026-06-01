"""
Start / heartbeat / stop email notifier for the AI Face Matching service.

Runs as its OWN small process, triggered by Task Scheduler - deliberately
separate from the main service, because the service is force-killed at 06:00 and
cannot run its own shutdown email.

IMPORTANT: 'start' and 'check' VERIFY the service is actually running before
they say so, so you never get a false "started" when the exe failed to launch.

Usage / schedule (all Daily):
    python notifier.py start    # ~02:05  -> "started OK" or "FAILED to start"
    python notifier.py check    # 03:00 / 04:00 / 05:00 -> heartbeat / alert
    python notifier.py stop     # ~06:02  -> night's summary

Strictly read-only with respect to the databases.
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Load the same .env the service uses (Task Scheduler's "Start in" should point
# at this folder, or run from here).
load_dotenv()

import mailer  # noqa: E402
import health  # noqa: E402


def _only_on_failure():
    # Default TRUE: healthy hourly heartbeats stay silent; we only email on the
    # start confirmation, the stop summary, and any DOWN alert. Set this to
    # "false" if you want a "Running OK" email every hour instead.
    return os.getenv("NOTIFY_HEARTBEAT_ONLY_ON_FAILURE", "true").lower() in (
        "1", "true", "yes", "y")


def _context_line():
    """Optional 'last log activity' hint, if NOTIFY_LOG_DIR is set."""
    age = health.newest_log_age_minutes()
    if age is None:
        return ""
    return f" Last log activity: {age:.0f} min ago."


def _send(subject, text, html=None):
    mailer.send_email(subject, text, html)


def _alert_down(now):
    subject = f"[ALERT] AI Face Matching is NOT running ({now})"
    text = (f"WARNING: the AI Face Matching service is NOT running at {now}, "
            f"but it is supposed to be running between 02:00 and 06:00.\n\n"
            f"Please check the server.{_context_line()}")
    html = (f"<html><body style='font-family:Segoe UI,Arial,sans-serif'>"
            f"<h2 style='color:#b00020;margin:0 0 4px'>&#9888; Service is NOT "
            f"running</h2><p>Checked at <b>{now}</b>. It should be running "
            f"between 02:00 and 06:00.</p><p>Please check the server.</p>"
            f"<p style='color:#666'>{_context_line()}</p></body></html>")
    _send(subject, text, html)


def _unknown(now):
    subject = f"AI Face Matching - status could not be verified ({now})"
    text = (f"Could not determine whether the service is running at {now} "
            f"(process check was inconclusive). No action may be needed, but "
            f"worth a quick look.{_context_line()}")
    _send(subject, text)


def notify_start():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    state = health.service_running()
    if state is True:
        subject = f"AI Face Matching - Started OK ({datetime.now():%Y-%m-%d})"
        text = (f"AI Face Matching service confirmed running at {now}.\n\n"
                f"It will process the queue and stop automatically at 06:00."
                f"{_context_line()}")
        html = (f"<html><body style='font-family:Segoe UI,Arial,sans-serif'>"
                f"<h2 style='color:#1a7f37;margin:0 0 4px'>&#9989; Service "
                f"started</h2><p>Confirmed running at <b>{now}</b>.</p>"
                f"<p style='color:#666'>Stops automatically at 06:00."
                f"{_context_line()}</p></body></html>")
        _send(subject, text, html)
    elif state is False:
        # The important case: it was supposed to start and didn't.
        subject = f"[ALERT] AI Face Matching FAILED to start ({now})"
        text = (f"WARNING: the AI Face Matching service did NOT start at {now}. "
                f"The scheduled 02:00 run does not appear to be running.\n\n"
                f"Please check the server.{_context_line()}")
        _send(subject, text)
    else:
        _unknown(now)


def notify_check():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    state = health.service_running()
    if state is True:
        if _only_on_failure():
            print("check: running and NOTIFY_HEARTBEAT_ONLY_ON_FAILURE set "
                  "-> no email.", flush=True)
            return
        subject = f"AI Face Matching - Running OK ({now})"
        text = (f"Heartbeat: service confirmed running at {now}. "
                f"Stops automatically at 06:00.{_context_line()}")
        html = (f"<html><body style='font-family:Segoe UI,Arial,sans-serif'>"
                f"<h2 style='color:#1a7f37;margin:0 0 4px'>&#9989; Running</h2>"
                f"<p>Confirmed at <b>{now}</b>.</p>"
                f"<p style='color:#666'>{_context_line()}</p></body></html>")
        _send(subject, text, html)
    elif state is False:
        _alert_down(now)
    else:
        _unknown(now)


def notify_stop():
    import summary  # imported here so start/check never need pyodbc/pandas
    try:
        data = summary.build_summary()
        subject, text, html = summary.render(data)
    except Exception as e:  # noqa: BLE001 - still send a "stopped" note
        subject = f"AI Face Matching - Service Stopped ({datetime.now():%Y-%m-%d})"
        text = ("AI Face Matching service has stopped.\n\n"
                f"(Could not build the summary: {type(e).__name__}: {e})")
        html = None
    _send(subject, text, html)


def main():
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "").strip()
    if mode == "start":
        notify_start()
    elif mode == "check":
        notify_check()
    elif mode == "stop":
        notify_stop()
    else:
        print("Usage: python notifier.py [start|check|stop]", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
