# Face Recognition Attendance System

A comprehensive Django-based attendance management system using face recognition technology. Students register with their face, name, and enrollment number, and attendance is automatically marked using real-time face recognition.

## Features

- **Student Registration**: Students register with name, enrollment number, and face capture using webcam
- **Face Recognition**: Advanced face recognition algorithm using `face_recognition` library with distance threshold for accuracy
- **Daily Attendance Logging**: Automatic attendance marking with present/absent status
- **Duplicate Prevention**: Prevents marking attendance twice for the same student on the same day
- **Real-time Statistics**: View attendance statistics including total students, present, absent, and attendance percentage
- **Modern UI**: Beautiful, responsive user interface with gradient designs

## Technology Stack

- **Backend**: Django 5.2+
- **Face Recognition**: face_recognition library (dlib-based)
- **Computer Vision**: OpenCV (cv2)
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite (default, can be changed to PostgreSQL/MySQL)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip
- Webcam for face capture

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd d:\working\Attendence
   ```

2. **Activate virtual environment** (if using one)
   ```bash
   # Windows PowerShell
   .\env\Scripts\Activate.ps1
   
   # Windows CMD
   .\env\Scripts\activate.bat
   
   # Linux/Mac
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Installing `dlib` and `face_recognition` may require additional system dependencies:
   - **Windows**: Visual Studio Build Tools
   - **Linux**: `cmake`, `libopenblas-dev`, `liblapack-dev`
   - **Mac**: Xcode Command Line Tools

4. **Run migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create superuser** (optional, for admin access)
   ```bash
   python manage.py createsuperuser
   ```

6. **Run development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   - Open browser and navigate to: `http://127.0.0.1:8000/`

## Usage

### 1. Register Student

1. Navigate to the home page
2. Click "Register Student"
3. Fill in:
   - Full Name
   - Enrollment Number
4. Allow webcam access when prompted
5. Position face in the camera frame
6. Click "Capture Face" to capture your face
7. Click "Register Student" to complete registration

### 2. Mark Attendance

1. Navigate to "Mark Attendance" page
2. Allow webcam access
3. Position face in the camera frame
4. Click "Recognize Face"
5. System will:
   - Detect and recognize the face
   - Match with registered students
   - Mark attendance automatically
   - Display confirmation with confidence score

### 3. View Attendance Records

1. Navigate to "View Attendance Records"
2. See statistics for today:
   - Total Students
   - Present Count
   - Absent Count
   - Attendance Percentage
3. View detailed attendance list with:
   - Student name and enrollment
   - Time of attendance
   - Status (Present/Absent)
   - Recognition confidence score

## Face Recognition Algorithm

The system uses an advanced face recognition algorithm with the following features:

- **Model**: Uses the 'large' model for face encoding (128-dimensional vectors)
- **Detection**: HOG (Histogram of Oriented Gradients) model for face detection
- **Distance Threshold**: 0.5 (lower = stricter matching)
- **Confidence Calculation**: Converts face distance to confidence score (0-1)
- **Validation**: 
  - Checks for single face in frame
  - Validates face quality
  - Prevents duplicate attendance

## Project Structure

```
Attendence/
├── app/
│   ├── models.py          # Student and Attendance models
│   ├── views.py           # View functions
│   ├── forms.py           # Django forms
│   ├── utils.py           # Face recognition utilities
│   ├── templates/         # HTML templates
│   │   ├── home.html
│   │   ├── register.html
│   │   ├── attendance.html
│   │   ├── registration_success.html
│   │   └── attendance_list.html
│   └── migrations/        # Database migrations
├── controller/
│   ├── settings.py        # Django settings
│   ├── urls.py            # URL routing
│   └── wsgi.py
├── manage.py
├── requirements.txt
└── README.md
```

## API Endpoints

- `GET /` - Home page
- `GET /register/` - Student registration page
- `POST /register/` - Submit registration
- `GET /attendance/` - Attendance marking page
- `POST /api/mark-attendance/` - API endpoint for marking attendance
- `GET /attendance-list/` - View attendance records
- `GET /admin/` - Django admin panel

## Database Models

### Student
- `name`: Student's full name
- `enrollment_number`: Unique enrollment identifier
- `face_encoding`: JSON-encoded face recognition vector
- `registered_at`: Registration timestamp
- `is_active`: Active status flag

### Attendance
- `student`: Foreign key to Student
- `date`: Attendance date
- `time`: Attendance time
- `status`: Present or Absent
- `confidence`: Face recognition confidence score
- **Unique constraint**: (student, date) - prevents duplicate attendance

## Troubleshooting

### Webcam Not Working
- Check browser permissions for camera access
- Ensure no other application is using the webcam
- Try a different browser (Chrome recommended)

### Face Recognition Not Working
- Ensure good lighting
- Face the camera directly
- Remove glasses/hats if possible
- Ensure face is clearly visible in frame

### Installation Issues
- For `dlib` installation issues, refer to: https://github.com/davisking/dlib
- For Windows, ensure Visual Studio Build Tools are installed
- For Linux, install system dependencies: `sudo apt-get install cmake libopenblas-dev liblapack-dev`

## Security Notes

- This is a development version. For production:
  - Change `DEBUG = False` in settings.py
  - Set a strong `SECRET_KEY`
  - Use a production database (PostgreSQL recommended)
  - Implement HTTPS
  - Add authentication/authorization
  - Secure face encoding data

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to submit issues and enhancement requests!
