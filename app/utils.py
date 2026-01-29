import face_recognition
import cv2
import numpy as np
import json
import base64
from PIL import Image
import io
from .models import Student, Attendance
import datetime
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string


def process_face_image_from_base64(image_data):
    """
    Process base64 encoded image and extract face encoding.
    Uses the best face recognition algorithm with HOG model for detection.
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to RGB if it's not already (handles RGBA, L, P, etc.)
        if image.mode != 'RGB':
            # Convert to RGB, removing alpha channel if present
            rgb_image_pil = image.convert('RGB')
        else:
            rgb_image_pil = image
        
        # Convert PIL image to numpy array (RGB) - ensure uint8
        rgb_image = np.array(rgb_image_pil, dtype=np.uint8)
        
        # Ensure it's a 3-channel RGB image
        if len(rgb_image.shape) != 3:
            return None, "Invalid image format. Image must be a color image (3 channels)."
        
        if rgb_image.shape[2] != 3:
            return None, f"Invalid image format. Expected 3 channels, got {rgb_image.shape[2]}."
        
        # Ensure image is contiguous in memory (required by face_recognition)
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
        
        # Final validation: ensure dtype is uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        
        # Use HOG model for face detection (faster and good accuracy)
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if not face_locations:
            return None, "No face detected. Please ensure your face is clearly visible."
        
        if len(face_locations) > 1:
            return None, "Multiple faces detected. Please ensure only one person is in the frame."
        
        # Extract face encoding using the best model
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=1, model='large')
        
        if not face_encodings:
            return None, "Could not generate face encoding. Please try again."
        
        # Convert to JSON string for storage
        encoding_list = face_encodings[0].tolist()
        encoding_json = json.dumps(encoding_list)
        
        return encoding_json, None
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def capture_face_from_webcam():
    """
    Capture face from webcam using OpenCV.
    Returns face encoding as JSON string or None if failed.
    """
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
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using HOG model
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        
        # Draw rectangle around face
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected - Press SPACE", (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Face Capture - Press SPACE to capture, Q to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Spacebar to capture
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
        
        elif key == ord('q'):  # Q to quit
            error = "Capture cancelled"
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    
    return face_encoding, error


def recognize_face_and_mark_attendance(image_data=None, video_capture=None):
    """
    Recognize face and mark attendance.
    Attendance is allowed only before closing time.
    """

    # # 🔒 ATTENDANCE TIME CHECK (NEW)
    # from .utils import is_attendance_open

    # if not is_attendance_open():
    #     return None, None, "Attendance is closed for today"

    # Load all active students
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

    # ---------------- IMAGE PROCESSING ----------------
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

    # ---------------- FACE DETECTION ----------------
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

    # ---------------- MATCHING ----------------
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

    # ---------------- MARK ATTENDANCE ----------------
    try:
        attendance = Attendance.objects.create(
            student=matched_student,
            date=today,
            time=datetime.datetime.now().time(),
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

from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string


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

from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.timezone import now

def send_absent_alert_email(student, date):
    try:
        subject = "Absent Alert | Face Attendance System"

        html_content = render_to_string(
            "emails/absent_alert_mail.html",
            {
                "student": student,
                "date": date,
                "year": now().year,
            }
        )

        text_content = f"You were marked ABSENT on {date}"

        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[settings.DEPARTMENT_EMAIL],
        )

        email.attach_alternative(html_content, "text/html")
        email.send()

    except Exception as e:
        print("Absent Mail Error:", e)

def send_daily_attendance_report():
    try:
        today = datetime.date.today()
        stats = get_attendance_stats(today)

        subject = f"Daily Attendance Report | {today}"

        html_content = render_to_string(
            "emails/daily_report_mail.html",
            {
                "stats": stats,
                "date": today,
                "year": now().year,
            }
        )

        text_content = (
            f"Total: {stats['total_students']}, "
            f"Present: {stats['present']}, "
            f"Absent: {stats['absent']}"
        )

        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[settings.DEPARTMENT_EMAIL],
        )

        email.attach_alternative(html_content, "text/html")
        email.send()

    except Exception as e:
        print("Daily Report Mail Error:", e)

# from django.conf import settings
# from datetime import datetime

# def is_attendance_open():
#     """
#     Returns True if attendance is still open, else False
#     """
#     closing_time = datetime.strptime(
#         settings.ATTENDANCE_CLOSING_TIME, "%H:%M"
#     ).time()

#     current_time = datetime.now().time()

#     return current_time <= closing_time
