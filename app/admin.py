from django.contrib import admin, messages
from django.db.models import Count
from django.urls import path, reverse
from django.http import HttpResponseRedirect
import datetime
from django.contrib import admin
from django.urls import path
from django.http import HttpResponseRedirect
from django.contrib import messages
from django.urls import reverse
import datetime
from django.db.models import Count
from .models import Attendance, Student
from .models import Student, Attendance

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['name', 'enrollment_number', 'registered_at', 'is_active']
    list_filter = ['is_active', 'registered_at']
    search_fields = ['name', 'enrollment_number']
    readonly_fields = ['registered_at']



@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'date', 'time', 'status', 'confidence']
    list_filter = ['date', 'status']
    search_fields = ['student__name', 'student__enrollment_number']
    date_hierarchy = 'date'
    readonly_fields = ['confidence']
    change_list_template = "admin/app/attendance/change_list.html"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "reset-today/",
                self.admin_site.admin_view(self.reset_today_view),
                name="app_attendance_reset_today",
            ),
        ]
        return custom_urls + urls

    def reset_today_view(self, request):
        """Reset today's attendance records."""
        if request.method != "POST":
            messages.error(request, "Invalid request method.")
            return HttpResponseRedirect("../")

        today = datetime.date.today()
        deleted, _ = Attendance.objects.filter(date=today).delete()
        messages.success(request, f"Reset complete. Deleted {deleted} record(s) for {today}.")
        return HttpResponseRedirect(reverse("admin:app_attendance_changelist"))

    def changelist_view(self, request, extra_context=None):
        """Override changelist view to include attendance summary."""
        today = datetime.date.today()

        qs = Attendance.objects.filter(date=today).values("status").annotate(c=Count("id"))
        counts = {row["status"]: row["c"] for row in qs}

        present = int(counts.get("Present", 0))
        absent = int(counts.get("Absent", 0))

        total_students = Student.objects.count()
        absent = total_students - present

        total = present + absent

        extra_context = extra_context or {}
        extra_context.update(
            {
                "attendance_chart_date": today,
                "attendance_present": present,
                "attendance_absent": absent,
                "attendance_total": total,
                "reset_today_url": reverse("admin:app_attendance_reset_today"),
            }
        )
        return super().changelist_view(request, extra_context=extra_context)
