import os
import django
from django.core.mail import send_mail
from decouple import config

# Configura el entorno de Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vigilancia.settings')
django.setup()

# Enviar el correo de prueba
try:
    send_mail(
        'Prueba de Envío de Correo',
        'Este es un correo de prueba enviado desde un script independiente de Django.',
        config('DEFAULT_FROM_EMAIL'),  # Correo remitente desde tu archivo .env
        ['ccarrielm@unemi.edu.ec'],  # Cambia esto por el correo al que quieras enviar la prueba
        fail_silently=False,
    )
    print("Correo enviado exitosamente.")
except Exception as e:
    print(f"Error al enviar el correo: {e}")
