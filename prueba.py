import os
import django
from django.core.mail import send_mail
from decouple import config




print("EMAIL_HOST_USER:", config('EMAIL_HOST_USER'))
