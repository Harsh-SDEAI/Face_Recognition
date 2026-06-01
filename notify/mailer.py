"""
Tiny SMTP sender for the start/stop notifications (Office 365 / Outlook).

Reads all connection settings from environment variables (loaded from the same
.env the service uses). Sends a multipart text+HTML email to one or more
recipients. Nothing here touches a database.

Required .env keys:
    SMTP_USER         e.g. alerts@yourcompany.com   (the mailbox to log in as)
    SMTP_PASSWORD     the password / app password for that mailbox
    NOTIFY_RECIPIENTS comma-separated list, e.g. "harsh.n@x.com, ops@x.com"

Optional .env keys (sensible Office 365 defaults):
    SMTP_HOST   default smtp.office365.com
    SMTP_PORT   default 587  (STARTTLS)
    SMTP_FROM   default = SMTP_USER
"""

import os
import smtplib
import ssl
from email.message import EmailMessage


def _recipients():
    raw = os.getenv("NOTIFY_RECIPIENTS", "")
    return [a.strip() for a in raw.split(",") if a.strip()]


def send_email(subject, text_body, html_body=None):
    """
    Send one email. Returns True on success, False on failure (never raises),
    so a mail problem can never crash the caller.
    """
    host = os.getenv("SMTP_HOST", "smtp.office365.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", user)
    recipients = _recipients()

    if not (user and password and recipients):
        print("MAILER: missing SMTP_USER / SMTP_PASSWORD / NOTIFY_RECIPIENTS in "
              ".env - cannot send.", flush=True)
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(user, password)
            server.send_message(msg)
        print(f"MAILER: sent '{subject}' to {len(recipients)} recipient(s).",
              flush=True)
        return True
    except Exception as e:  # noqa: BLE001 - report and move on, never crash
        print(f"MAILER: failed to send '{subject}': {type(e).__name__}: {e}",
              flush=True)
        return False
