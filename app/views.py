from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import datetime
from calendar import monthrange
from datetime import date as date_class
from app.utils import get_attendance_stats, is_attendance_closed
from django.conf import settings
from .models import Student, Attendance
from .forms import StudentForm
from .utils import process_face_image_from_base64, recognize_face_and_mark_attendance, get_attendance_stats
from django.utils.timezone import now
from django.http import HttpResponse
from .models import Attendance
from .utils import generate_professional_attendance_pdf


def register_student(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():
            student_name = form.cleaned_data['name']
            student_enrollment_number = form.cleaned_data['enrollment_number']
            captured_image = form.cleaned_data.get('captured_image')

            if Student.objects.filter(enrollment_number=student_enrollment_number).exists():
                return render(request, 'register.html', {
                    'form': form,
                    'error': 'Enrollment number already exists!'
                })

            if not captured_image:
                return render(request, 'register.html', {
                    'form': form,
                    'error': 'Please capture your face image!'
                })

            face_encoding, error = process_face_image_from_base64(captured_image)
            
            if error:
                return render(request, 'register.html', {
                    'form': form,
                    'error': error
                })

            try:
                student = Student.objects.create(
                    name=student_name,
                    enrollment_number=student_enrollment_number,
                    face_encoding=face_encoding
                )
                return redirect('registration_success', student_id=student.id)
            except Exception as e:
                return render(request, 'register.html', {
                    'form': form,
                    'error': f'Error creating student: {str(e)}'
                })
    else:
        form = StudentForm()

    return render(request, 'register.html', {'form': form})


def registration_success(request, student_id):
    try:
        student = Student.objects.get(id=student_id)
        return render(request, 'registration_success.html', {'student': student})
    except Student.DoesNotExist:
        return redirect('register')



def attendance_page(request):
    stats = get_attendance_stats()

    return render(request, "attendance.html", {
        "stats": stats})



@csrf_exempt
@require_http_methods(["POST"])
def mark_attendance_api(request):

    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        
        if not image_data:
            return JsonResponse({
                'success': False,
                'error': 'No image provided'
            }, status=400)
        
        student, confidence, error = recognize_face_and_mark_attendance(image_data=image_data)
        
        if error:
            return JsonResponse({
                'success': False,
                'error': error,
                'student_name': student.name if student else None,
                'confidence': float(confidence) if confidence else None
            })
        
        stats = get_attendance_stats()
        return JsonResponse({
            'success': True,
            'student_name': student.name,
            'enrollment_number': student.enrollment_number,
            'confidence': float(confidence),
            'message': f'Attendance marked successfully for {student.name}',
            'stats': stats
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, status=500)


def attendance_list(request):
    date_param = request.GET.get('date')

    try:
        date_obj = (
            datetime.datetime.strptime(date_param, '%Y-%m-%d').date()
            if date_param else None
        )
    except ValueError:
        date_obj = None

    stats = get_attendance_stats(date_obj)

    all_students = Student.objects.filter(is_active=True).order_by(
        "enrollment_number"
    )

    attendances = (
        Attendance.objects
        .filter(date=stats['date'])
        .select_related('student')
        .order_by("student__enrollment_number")
    )

    present_ids = attendances.filter(
        status='Present'
    ).values_list('student_id', flat=True)

    present_students = Student.objects.filter(
        id__in=present_ids
    ).order_by("enrollment_number")

    absent_students = Student.objects.filter(
        is_active=True
    ).exclude(
        id__in=present_ids
    ).order_by("enrollment_number")

    context = {
        'attendances': attendances,
        'present_students': present_students,
        'absent_students': absent_students,
        'stats': stats
    }

    return render(request, 'attendance_list.html', context)


def home(request):
    stats = get_attendance_stats()
    attendance_closed = is_attendance_closed()

    return render(request, "home.html", {
        "stats": stats,
        "attendance_closed": attendance_closed,
        "closing_time": settings.ATTENDANCE_CLOSING_TIME
    })


@require_http_methods(["POST"])
def mark_absent_today_api(request):
    today = datetime.date.today()
    students = Student.objects.filter(is_active=True)

    created = 0
    for student in students:
        if not Attendance.objects.filter(student=student, date=today).exists():
            Attendance.objects.create(
                student=student,
                date=today,
                time=datetime.datetime.now().time(),
                status="Absent",
                confidence=None,
            )
            created += 1

    stats = get_attendance_stats(today)
    return JsonResponse(
        {
            "success": True,
            "message": f"Marked Absent for {created} student(s).",
            "created": created,
            "stats": stats,
        }
    )


@require_http_methods(["POST"])
def reset_attendance_today_api(request):
    today = datetime.date.today()
    deleted, _ = Attendance.objects.filter(date=today).delete()

    stats = get_attendance_stats(today)
    return JsonResponse(
        {
            "success": True,
            "message": f"Reset complete. Deleted {deleted} attendance record(s) for today.",
            "deleted": deleted,
            "stats": stats,
        }
    )
@require_http_methods(["POST"])
def send_daily_report_api(request):
    from .utils import send_daily_attendance_report
    send_daily_attendance_report()
    return JsonResponse({"success": True, "message": "Daily report sent"})


def download_daily_report(request):
    date = now().date()

    students = Student.objects.filter(is_active=True).order_by(
        "enrollment_number"
    )

    attendance_map = {
        a.student_id: a.status
        for a in Attendance.objects.filter(date=date)
    }

    table_data = [
        ["Enrollment No", "Student Name", "Status"]
    ]

    for student in students:
        status = attendance_map.get(student.id, "Absent")

        table_data.append([
            student.enrollment_number,
            student.name,
            status
        ])

    pdf = generate_professional_attendance_pdf(
        title="Daily Attendance Report",
        subtitle=f"Date: {date.strftime('%d %B %Y')}",
        table_data=table_data
    )

    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="daily_report_{date}.pdf"'
    )
    return response




def download_monthly_summary_report(request, year, month):
    total_days = monthrange(year, month)[1]

    students = Student.objects.filter(
        is_active=True
    ).order_by("enrollment_number")

    table_data = [
        ["Enrollment No", "Student Name", "Present Days", "Absent Days"]
    ]

    for student in students:
        present_days = Attendance.objects.filter(
            student=student,
            date__year=year,
            date__month=month,
            status="Present"
        ).count()

        absent_days = total_days - present_days

        table_data.append([
            student.enrollment_number,
            student.name,
            present_days,
            absent_days
        ])

    pdf = generate_professional_attendance_pdf(
        title="Monthly Attendance Summary Report",
        subtitle=f"Month: {month} / {year}",
        table_data=table_data
    )

    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="monthly_attendance_{month}_{year}.pdf"'
    )
    return response


def download_datewise_report(request, year, month, day):
    selected_date = date_class(year, month, day)

    students = Student.objects.filter(
        is_active=True
    ).order_by("enrollment_number")

    attendance_map = {
        a.student_id: a.status
        for a in Attendance.objects.filter(date=selected_date)
    }

    table_data = [
        ["Enrollment No", "Student Name", "Status"]
    ]

    for student in students:
        status = attendance_map.get(student.id, "Absent")

        table_data.append([
            student.enrollment_number,
            student.name,
            status
        ])

    pdf = generate_professional_attendance_pdf(
        title="Daily Attendance Report",
        subtitle=f"Date: {selected_date.strftime('%d %B %Y')}",
        table_data=table_data
    )

    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="daily_report_{selected_date}.pdf"'
    )
    return response
