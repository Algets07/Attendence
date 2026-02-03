from django.db import models
from django.core.exceptions import ValidationError
import json



class Student(models.Model):
    name = models.CharField(max_length=100)
    enrollment_number = models.CharField(max_length=20, unique=True)
    face_encoding = models.TextField()  
    registered_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.enrollment_number})"
    
    def get_face_encoding(self):
        """Convert stored JSON string back to list"""
        try:
            return json.loads(self.face_encoding)
        except:
            return None

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    time = models.TimeField()
    status = models.CharField(max_length=10, choices=[('Present', 'Present'), ('Absent', 'Absent')], default='Present')
    confidence = models.FloatField(null=True, blank=True, help_text="Face recognition confidence score")

    class Meta:
        unique_together = ['student', 'date']  
        ordering = ['-date', '-time']

    def __str__(self):
        return f"{self.student.name} - {self.date} - {self.status}"

