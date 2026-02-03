# def close_attendance_and_send_report():
#     today = now().date()

#     # 1️⃣ Mark absentees
#     students = Student.objects.filter(is_active=True)
#     created = 0

#     for student in students:
#         if not Attendance.objects.filter(student=student, date=today).exists():
#             Attendance.objects.create(
#                 student=student,
#                 date=today,
#                 time=now().time(),
#                 status="Absent",
#                 confidence=None,
#             )
#             created += 1
#     current_time = localtime().time().strftime("%H.%M")
#     closing_time = settings.ATTENDANCE_CLOSING_TIME.strftime("%H.%M")
#     if current_time== closing_time:
#             print(current_time)
#             send_daily_attendance_report()

#     return created