from django.core.management.base import BaseCommand
from app.utils import close_attendance_and_send_report


class Command(BaseCommand):
    help = "Close attendance and send daily attendance report"

    def handle(self, *args, **kwargs):
        close_attendance_and_send_report()
        self.stdout.write(
            self.style.SUCCESS("Attendance closed & daily report sent")
        )
