# surveillance/services/notification_service.py
from django.core.mail import send_mail
from .models import Alert

def send_alert(user, message, severity):
    Alert.objects.create(user=user, message=message, severity_level=severity)
    send_mail(
        subject=f'Alert: {severity} Level Detected',
        message=message,
        from_email='no-reply@detectionsystem.com',
        recipient_list=[user.email],
        fail_silently=False,
    )
