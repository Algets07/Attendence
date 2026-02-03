import face_recognition
import cv2
import numpy as np
import json
import base64
from PIL import Image
import io
from .models import Student, Attendance
import datetime
from django.conf import settings
from django.template.loader import render_to_string
from django.core.mail import EmailMultiAlternatives
from django.utils.timezone import now
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from django.http import HttpResponse
import io



def process_face_image_from_base64(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            rgb_image_pil = image.convert('RGB')
        else:
            rgb_image_pil = image
        
        rgb_image = np.array(rgb_image_pil, dtype=np.uint8)
        
        if len(rgb_image.shape) != 3:
            return None, "Invalid image format. Image must be a color image (3 channels)."
        
        if rgb_image.shape[2] != 3:
            return None, f"Invalid image format. Expected 3 channels, got {rgb_image.shape[2]}."
        
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
        
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if not face_locations:
            return None, "No face detected. Please ensure your face is clearly visible."
        
        if len(face_locations) > 1:
            return None, "Multiple faces detected. Please ensure only one person is in the frame."
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=1, model='large')
        
        if not face_encodings:
            return None, "Could not generate face encoding. Please try again."
        
        encoding_list = face_encodings[0].tolist()
        encoding_json = json.dumps(encoding_list)
        
        return encoding_json, None
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def capture_face_from_webcam():
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        return None, "Could not open webcam"
    
    print("Capturing face... Press SPACE to capture, Q to quit")
    
    face_encoding = None
    error = None
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            error = "Failed to read from webcam"
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected - Press SPACE", (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Face Capture - Press SPACE to capture, Q to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '): 
            if face_locations:
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, face_locations, num_jitters=1, model='large'
                )
                if face_encodings:
                    encoding_list = face_encodings[0].tolist()
                    face_encoding = json.dumps(encoding_list)
                    print("Face captured successfully!")
                    break
                else:
                    error = "Could not generate face encoding"
            else:
                error = "No face detected. Please position your face in the frame."
        
        elif key == ord('q'):  
            error = "Capture cancelled"
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    
    return face_encoding, error

from django.utils.timezone import localtime
def recognize_face_and_mark_attendance(image_data=None, video_capture=None):
    current_time = localtime().time()
    if current_time > settings.ATTENDANCE_CLOSING_TIME:
        return None, None, "Attendance time is closed"
    students = Student.objects.filter(is_active=True)

    if not students.exists():
        return None, None, "No registered students found"

    known_face_encodings = []
    known_students = []

    for student in students:
        encoding = student.get_face_encoding()
        if encoding:
            known_face_encodings.append(np.array(encoding))
            known_students.append(student)

    if not known_face_encodings:
        return None, None, "No valid face encodings found in database"

    if image_data:
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            rgb_image = np.array(image, dtype=np.uint8)

            if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
                return None, None, "Invalid image format"

            if not rgb_image.flags['C_CONTIGUOUS']:
                rgb_image = np.ascontiguousarray(rgb_image)

        except Exception as e:
            return None, None, f"Error processing image: {str(e)}"

    elif video_capture:
        ret, frame = video_capture.read()
        if not ret:
            return None, None, "Failed to read from webcam"
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    else:
        return None, None, "No image source provided"

    face_locations = face_recognition.face_locations(rgb_image, model='hog')

    if not face_locations:
        return None, None, "No face detected"

    if len(face_locations) > 1:
        return None, None, "Multiple faces detected"

    face_encodings = face_recognition.face_encodings(
        rgb_image, face_locations, num_jitters=1, model='large'
    )

    if not face_encodings:
        return None, None, "Could not generate face encoding"

    face_encoding = face_encodings[0]

    face_distances = face_recognition.face_distance(
        known_face_encodings, face_encoding
    )

    best_match_index = np.argmin(face_distances)
    best_distance = face_distances[best_match_index]

    TOLERANCE = 0.5

    if best_distance > TOLERANCE:
        return None, None, "Face not recognized"

    matched_student = known_students[best_match_index]
    confidence = 1 - best_distance

    today = datetime.date.today()

    if Attendance.objects.filter(student=matched_student, date=today).exists():
        return matched_student, confidence, "Attendance already marked today"

    try:
        attendance = Attendance.objects.create(
            student=matched_student,
            date=today,
            time=localtime().time(),
            status="Present",
            confidence=confidence
        )

        send_attendance_email(matched_student, attendance)

        return matched_student, confidence, None

    except Exception as e:
        return matched_student, confidence, f"Error marking attendance: {str(e)}"

def get_attendance_stats(date=None):
    if date is None:
        date = datetime.date.today()

    total_students = Student.objects.filter(is_active=True).count()
    present_count = Attendance.objects.filter(date=date, status='Present').count()
    absent_count = total_students - present_count

    return {
        'date': date,
        'total_students': total_students,
        'present': present_count,
        'absent': absent_count,
        'attendance_percentage': (
            (present_count / total_students) * 100 if total_students > 0 else 0
        )
    }

def send_attendance_email(student, attendance):
    try:
        department_email = getattr(
            settings, 'DEPARTMENT_EMAIL', 'tamilarasan6112002@gmail.com'
        )

        subject = f"Attendance Confirmed | {student.name}"

        confidence = f"{attendance.confidence * 100:.1f}" if attendance.confidence else "N/A"

        html_content = render_to_string(
            "emails/attendance_mail.html",
            {
                "student": student,
                "attendance": attendance,
                "confidence": confidence,
                "year": now().year,
            }
        )

        text_content = (
            f"Attendance confirmed for {student.name} "
            f"({student.enrollment_number}) on {attendance.date}"
        )

        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[department_email],
        )

        email.attach_alternative(html_content, "text/html")
        email.send()

    except Exception as e:
        print("Email Error:", e)


def send_daily_attendance_report():
    date = now().date()

    present_list = Attendance.objects.filter(
        date=date,
        status="Present"
    ).select_related("student")

    present_student_ids = present_list.values_list("student_id", flat=True)

    absent_list = Student.objects.filter(
        is_active=True
    ).exclude(id__in=present_student_ids)

    stats = {
        "total_students": Student.objects.filter(is_active=True).count(),
        "present": present_list.count(),
        "absent": absent_list.count(),
        "attendance_percentage": (
            (present_list.count() /
             Student.objects.filter(is_active=True).count()) * 100
            if Student.objects.filter(is_active=True).exists() else 0
        )
    }

    html_content = render_to_string(
        "emails/daily_report_mail.html",
        {
            "date": date.strftime("%B %d, %Y"),
            "stats": stats,
            "present_list": present_list,
            "absent_list": absent_list,
            "year": now().year
        }
    )

    email = EmailMultiAlternatives(
        subject=f"Daily Attendance Report | {date}",
        body="Daily attendance report attached.",
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=[settings.DEPARTMENT_EMAIL],
    )

    email.attach_alternative(html_content, "text/html")
    email.send()

from django.utils.timezone import now

def close_attendance_and_send_report():
    today = now().date()

    # 1️⃣ Mark absentees
    students = Student.objects.filter(is_active=True)
    created = 0

    for student in students:
        if not Attendance.objects.filter(student=student, date=today).exists():
            Attendance.objects.create(
                student=student,
                date=today,
                time=now().time(),
                status="Absent",
                confidence=None,
            )
            created += 1

    # 2️⃣ Send daily report
    send_daily_attendance_report()

    return created



def is_attendance_closed():
    current_time = localtime().time()
    tem_curtime = f'{str(current_time)[:5]}'

    set_time = settings.ATTENDANCE_CLOSING_TIME
    sys_curtime = f'{str(set_time)[:5]}'

    print('testing time is :',tem_curtime)
    # print("NOW:", localtime().time())
    print("CLOSE:",sys_curtime)
    if tem_curtime ==sys_curtime:
        print('hi sent')
        close_attendance_and_send_report()
    return current_time >= settings.ATTENDANCE_CLOSING_TIME
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
import io
def generate_professional_attendance_pdf(
    title,
    subtitle,
    table_data
):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []


    # ---- Report Title ----
    elements.append(Paragraph(
        title,
        ParagraphStyle(
            name="MainTitle",
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=10,
            textColor=colors.darkblue
        )
    ))

    elements.append(Paragraph(
        subtitle,
        ParagraphStyle(
            name="SubTitle",
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=20,
            textColor=colors.black
        )
    ))

    # ---- Table ----
    table = Table(table_data, repeatRows=1, colWidths=[100, 200, 100])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),

        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONT", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),

        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    elements.append(table)

    elements.append(Spacer(1, 20))

    # ---- Footer ----
    elements.append(Paragraph(
        "Generated by Attendance Management System",
        ParagraphStyle(
            name="Footer",
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

