from celery import shared_task
from .utils import close_attendance_and_send_report


@shared_task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 10})
def close_attendance_task(self):
    """
    Closes attendance, marks absentees,
    and sends daily attendance report
    """
    close_attendance_and_send_report()
    return "Attendance closed & report sent"
