from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings 

    
class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=13, unique=True)

    def __str__(self):
        return self.username

class Video(models.Model):
    file_path = models.CharField(max_length=255, unique=True) 
    uploaded_at = models.DateTimeField(auto_now_add=True)     

    def __str__(self):
        return self.file_path

class Detection(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE) 
    detected_at = models.DateTimeField(auto_now_add=True)       
    result = models.CharField(max_length=50)                    
    confidence = models.FloatField()                            

    def __str__(self):
        return f"{self.result} en {self.video.file_path} con {self.confidence*100}% de confianza"
    
class Incident(models.Model):
    INCIDENT_TYPES = [
        ('VIOLENCE', 'Violence'),
        ('NON_VIOLENCE', 'Non-Violence'),
    ]
    SOURCES = [
        ('REAL_TIME', 'Tiempo Real'),
        ('UPLOAD', 'Upload'),
    ]

    incident_type = models.CharField(max_length=50, choices=INCIDENT_TYPES, default='NON_VIOLENCE')
    location = models.CharField(max_length=255, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True, null=True)
    captured_video_url = models.URLField(blank=True, null=True)
    captured_image = models.ImageField(upload_to='incident_images/', blank=True, null=True)
    source = models.CharField(max_length=10, choices=SOURCES, default='REAL_TIME')  # Campo para identificar la fuente

    def __str__(self):
        return f"{self.incident_type} at {self.timestamp}"


class Alert(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    message = models.TextField()
    severity_level = models.CharField(max_length=50, choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')])
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.severity_level} Alert for {self.user}"
    

class IncidentReport(models.Model):
    detected_time = models.DateTimeField()
    location = models.CharField(max_length=255)
    severity = models.CharField(max_length=50)
    description = models.TextField()
    video_file = models.FileField(upload_to='incident_videos/')

    def __str__(self):
        return f"Incident on {self.detected_time} - {self.severity}"
    



