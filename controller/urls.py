"""
URL configuration for controller project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app.views import *



urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('register/', register_student, name='register'),
    path('registration-success/<int:student_id>/', registration_success, name='registration_success'),
    path('attendance/', attendance_page, name='attendance'),
    path('api/mark-attendance/', mark_attendance_api, name='mark_attendance_api'),
    path('api/mark-absent-today/', mark_absent_today_api, name='mark_absent_today_api'),
    path('api/reset-attendance-today/', reset_attendance_today_api, name='reset_attendance_today_api'),
    path('attendance-list/', attendance_list, name='attendance_list'),
    path("send-daily-report/", send_daily_report_api,name='daily'),
    path("download/daily/", download_daily_report, name="download_daily"),
    path("download/monthly-summary/<int:year>/<int:month>/",download_monthly_summary_report,name="download_monthly_summary"),
    path("download/datewise/<int:year>/<int:month>/<int:day>/",download_datewise_report,name="download_datewise_report"),


]
