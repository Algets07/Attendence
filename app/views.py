from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import datetime
from .models import Student, Attendance
from .forms import StudentForm
from .utils import process_face_image_from_base64, recognize_face_and_mark_attendance, get_attendance_stats


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
    
    date = request.GET.get('date')
    
    if date:
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            date_obj = None
    else:
        date_obj = None

    stats = get_attendance_stats(date_obj)
    
    all_students = Student.objects.all()

    attendances = Attendance.objects.filter(date=stats['date']).select_related('student')

    present_students = [attendance.student for attendance in attendances if attendance.status == 'Present']
    absent_students = [student for student in all_students if student not in present_students]

    context = {
        'attendances': attendances,
        'present_students': present_students,
        'absent_students': absent_students,
        'stats': stats
    }

    return render(request, 'attendance_list.html', context)



def home(request):
    return render(request, "home.html")

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
