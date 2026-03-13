from logger_setup import init_logger
init_logger()
from AIPhotoMatch import process_tasks_once
from apscheduler.schedulers.blocking import BlockingScheduler
from settings import TASK_RUN_INTERVAL_MINUTES
from datetime import datetime, timedelta
import pyodbc
import settings
import traceback
import logging
scheduler = BlockingScheduler()

# Add job with immediate execution + scheduled interval
# scheduler.add_job(
#     func=process_tasks_once,
#     trigger='interval',
#     minutes=TASK_RUN_INTERVAL_MINUTES,
#     next_run_time=datetime.now() + timedelta(seconds=1),  
#     id="face_matching_task",
#     max_instances=1,
#     coalesce=True
# )

def scheduled_task_wrapper():
    try:
        logging.info("🕒 Scheduled face matching task triggered.")
        # Run your main logic
        process_tasks_once()

    except Exception as e:
        logging.error("🔥 Error in scheduled_task_wrapper:")
        logging.error(e, exc_info=True)

scheduler.add_job(
    func=scheduled_task_wrapper,
    trigger='interval',
    minutes=TASK_RUN_INTERVAL_MINUTES,
    next_run_time=datetime.now() + timedelta(seconds=1),
    id="face_matching_task",
    max_instances=settings.INSTANCE,
    coalesce=True
)

if __name__ == "__main__":
    print(f"🚀 Starting Face Matching Scheduler (every {TASK_RUN_INTERVAL_MINUTES} minutes)", flush=True)
    scheduler.start()