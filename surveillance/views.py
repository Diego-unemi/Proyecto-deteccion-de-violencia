# import cv2
# import time
# import threading
# from django.http import StreamingHttpResponse
# from surveillance.services.deteccion_servicio import DeteccionViolenciaService
# from surveillance.utils.video_utils import iniciar_camara, leer_frame
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from .forms import ContactForm

def home_view(request):
    return render(request, 'home.html') 

def about_view(request):
    return render(request, 'about.html') 

def dashboard_view(request):
    return render(request, 'dashboard.html')

def video_view(request):
    return render(request, 'video.html') 

def camera_view(request):
    return render(request, "camera.html")

def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            message = form.cleaned_data['message']

            send_mail(
                f'Mensaje de {name}',
                message,
                email,
                [settings.DEFAULT_FROM_EMAIL],
                fail_silently=False,
            )
            return redirect('home')
    else:
        form = ContactForm()

    return render(request, 'contact.html', {'form': form})

from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from .forms import ContactForm
from .models import Incident  # Importar el modelo de Incidente

# Importaciones necesarias
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from django.http import StreamingHttpResponse
from django.views.decorators import gzip

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("models\modelo_cnn_lstm.h5")

# Configuraciones
IMG_SIZE = 64
SEQ_LENGTH = 15
UMBRAL_INICIAL = 0.5
UMBRAL_MOVIMIENTO = 20000
frames_deque = deque(maxlen=SEQ_LENGTH)
predicciones = deque(maxlen=5)
frame_anterior = None
transmitir = False
# Generador de frames para la detección en tiempo real
def generar_frames():
    global frame_anterior, frames_deque, predicciones, transmitir
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("No se pudo acceder a la cámara.")

    while transmitir:
        ret, frame = cap.read()
        if not ret:
            break

        # Zoom para capturar más detalles a distancia
        h, w, _ = frame.shape
        crop_size = int(min(h, w) * 0.8)
        y1, x1 = (h - crop_size) // 2, (w - crop_size) // 2
        frame_cropped = frame[y1:y1 + crop_size, x1:x1 + crop_size]
        frame_resized = cv2.resize(frame_cropped, (IMG_SIZE, IMG_SIZE))

        # Preprocesar el frame
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
        frames_deque.append(frame_rgb)

        # Detección de movimiento
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_anterior is not None:
            diff = cv2.absdiff(frame_anterior, gray_frame)
            movimiento = np.sum(diff)

            if movimiento > UMBRAL_MOVIMIENTO and len(frames_deque) == SEQ_LENGTH:
                frames_array = np.array(list(frames_deque)).reshape(1, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
                prediccion = modelo.predict(frames_array)[0][0]
                predicciones.append(prediccion)

                prediccion_suavizada = np.median(predicciones)
                label = "Comportamiento violento" if prediccion_suavizada >= UMBRAL_INICIAL else "Comportamiento no violento"
                color = (0, 0, 255) if prediccion_suavizada >= UMBRAL_INICIAL else (0, 255, 0)

                # Visualización en pantalla
                cv2.putText(frame, f"{label}: {prediccion_suavizada:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (10, 60), (int(prediccion_suavizada * 300) + 10, 80), color, -1)
                cv2.putText(frame, f"Confianza: {prediccion_suavizada:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # Registro del incidente en la base de datos
                if prediccion_suavizada >= UMBRAL_INICIAL:
                    new_incident = Incident(
                        incident_type='VIOLENCE',
                        location="Cámara en tiempo real",
                        description=f"Predicción de VIOLENCE con confianza de {prediccion_suavizada:.2f}"
                    )
                    new_incident.save()

        frame_anterior = gray_frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()
    cv2.destroyAllWindows()

# Vista de la transmisión en tiempo real
@gzip.gzip_page
def video_feed(request):
    global transmitir
    transmitir = True
    return StreamingHttpResponse(generar_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def stop_feed(request):
    global transmitir
    transmitir = False
    return redirect('camera_view')

from django.shortcuts import render
from .models import Incident

def incident_list(request):
    incidents = Incident.objects.all().order_by('-timestamp')
    return render(request, 'incident_list.html', {'incidents': incidents})


from django.shortcuts import get_object_or_404, redirect
from .models import Incident

def delete_incident(request, id):
    incident = get_object_or_404(Incident, id=id)
    if request.method == 'POST':
        incident.delete()
        return redirect('incident_list')
