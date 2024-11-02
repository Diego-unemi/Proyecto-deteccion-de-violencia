from django.db import models

# Create your models here.
from django.db import models

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

    incident_type = models.CharField(max_length=50, choices=INCIDENT_TYPES, default='NON_VIOLENCE')
    location = models.CharField(max_length=255, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True, null=True)
    captured_video_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return f"{self.incident_type} at {self.timestamp}"
