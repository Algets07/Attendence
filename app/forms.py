from django import forms
from .models import Student

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ['name', 'enrollment_number']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your full name',
                'required': True
            }),
            'enrollment_number': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your enrollment number',
                'required': True
            })
        }
        labels = {
            'name': 'Full Name',
            'enrollment_number': 'Enrollment Number'
        }

    captured_image = forms.CharField(widget=forms.HiddenInput(), required=False)
    
    def clean_enrollment_number(self):
        enrollment_number = self.cleaned_data.get('enrollment_number')
        if enrollment_number:
            enrollment_number = enrollment_number.strip().upper()
        return enrollment_number
