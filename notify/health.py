"""
Is the service actually running?  (no false alarms)

The start/heartbeat emails must reflect REALITY, so before we ever say
"running" we check that the service process actually exists. On Windows this is
a plain `tasklist` lookup - no extra dependency, no database, no writes.

service_running() returns:
    True   -> the process is present
    False  -> the process is NOT present
    None   -> we couldn't determine it (so we won't cry wolf)
"""

import os
import subprocess


def service_running(image_name=None):
    image_name = image_name or os.getenv("SERVICE_PROCESS_NAME",
                                         "AIPhotoMatch2026.exe")
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {image_name}"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return None  # tasklist unavailable / errored -> unknown, don't alarm

    text = (out.stdout or "")
    # tasklist prints "INFO: No tasks ..." when nothing matches.
    if image_name.lower() in text.lower():
        return True
    if "no tasks" in text.lower():
        return False
    return None


def newest_log_age_minutes(log_dir=None):
    """
    Optional freshness hint: minutes since the most recently modified log file.
    Returns None if no log dir is configured or it can't be read. Purely
    informational - we do NOT treat 'quiet' as failure (the queue may be empty).
    """
    log_dir = log_dir or os.getenv("NOTIFY_LOG_DIR")
    if not log_dir or not os.path.isdir(log_dir):
        return None
    try:
        import time
        newest = None
        for name in os.listdir(log_dir):
            p = os.path.join(log_dir, name)
            if os.path.isfile(p):
                m = os.path.getmtime(p)
                newest = m if newest is None or m > newest else newest
        if newest is None:
            return None
        return round((time.time() - newest) / 60.0, 1)
    except Exception:
        return None
