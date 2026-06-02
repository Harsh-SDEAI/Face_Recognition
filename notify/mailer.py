"""
Tiny SMTP sender for the start/stop notifications.

Reads all connection settings from environment variables (loaded from the .env
in this folder). Sends a multipart text+HTML email to one or more recipients.
Nothing here touches a database.

Required .env keys:
    SMTP_USER         e.g. masterlysocial@gmail.com
    SMTP_PASSWORD     the app password for that mailbox
    NOTIFY_RECIPIENTS comma-separated list, e.g. "harsh.n@x.com, ops@x.com"

Optional .env keys (Gmail defaults):
    SMTP_HOST   default smtp.gmail.com
    SMTP_PORT   default 465  (direct SSL)
    SMTP_FROM   default = SMTP_USER

Port behaviour:
    465  -> SMTP_SSL  (direct SSL, used by Gmail)
    587  -> SMTP     + STARTTLS (used by Office 365)
    other -> same as 587
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
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "465"))
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

    ctx = ssl.create_default_context()
    try:
        if port == 465:
            # Direct SSL - Gmail and most modern providers on port 465
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=30) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            # STARTTLS - Office 365 and others on port 587
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=ctx)
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
