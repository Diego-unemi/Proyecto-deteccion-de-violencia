import pandas as pd
import plotly.express as px
from .models import IncidentReport
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from .forms import ContactForm
from .services.deteccion_servicio import generar_frames
from .models import Incident
from django.contrib.auth.decorators import permission_required, login_required
from django.conf import settings
from django.contrib.auth import login, authenticate, logout
from .forms import CustomUserCreationForm, CustomAuthenticationForm
from django.contrib import messages
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from django.shortcuts import get_object_or_404
import threading
from datetime import datetime
from django.core.mail import EmailMessage
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from .forms import VideoUploadForm
from collections import deque
import os


@login_required
def home_view(request):
    return render(request, 'home.html') 
@login_required
def about_view(request):
    return render(request, 'about.html') 
@login_required
def dashboard_view(request):
    return render(request, 'dashboard.html')
@login_required
def video_view(request):
    return render(request, 'video.html') 
@login_required
def camera_view(request):
    return render(request, "modulos/deteccion_tiemporeal_o_Importacion/tiemporeal.html")
@login_required
def modulos_view(request):
    return render(request, 'modulos/modulos.html')
@login_required
def module_select(request):
    return render(request, 'modulos/deteccion_tiemporeal_o_Importacion/sub_modulo_home.html')
@login_required
def video_upload(request):
    return render(request, 'modulos/deteccion_tiemporeal_o_Importacion/video_importado.html')
@login_required
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




# Cargar el modelo de detección de violencia previamente entrenado
modelo = tf.keras.models.load_model("models/modelo_cnn_lstm.h5")
IMG_SIZE = 64
SEQ_LENGTH = 15
UMBRAL_INICIAL = 0.95
UMBRAL_MOVIMIENTO = 20000
frames_deque = deque(maxlen=SEQ_LENGTH)
predicciones = deque(maxlen=5)
frame_anterior = None
transmitir = False
FRAME_SKIP = 5  # Controla la frecuencia de análisis

@login_required
def generar_frames(request):
    global frame_anterior, frames_deque, predicciones, transmitir
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("No se pudo acceder a la cámara.")

    frame_counter = 0

    while transmitir:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue  # Omite la predicción en este frame

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

                # Registro del incidente y envío de correo solo si la predicción es >= UMBRAL_INICIAL
                if prediccion_suavizada >= UMBRAL_INICIAL:
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_content = ContentFile(buffer.tobytes(), name=f'incident_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')

                    # Crear y guardar el incidente
                    new_incident = Incident(
                        incident_type='VIOLENCE',
                        location="Cámara en tiempo real",
                        description=f"Predicción de VIOLENCE con confianza de {prediccion_suavizada:.2f}",
                        source='REAL_TIME'
                    )
                    new_incident.captured_image.save(image_content.name, image_content, save=True)

                    # Enviar un correo en un hilo separado
                    threading.Thread(target=enviar_correo, args=(prediccion_suavizada, buffer, image_content, request)).start()

        frame_anterior = gray_frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()
    cv2.destroyAllWindows()

@login_required
def analizar_video_importado(request):
    resultado = None
    detalles = None
    video_path = None

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data['video_file']
            video_name = form.cleaned_data['video_name']
            video_description = form.cleaned_data['video_description']

            # Guardar el archivo de video en el servidor
            fs = FileSystemStorage()
            filename = fs.save(video_file.name, video_file)
            video_path = fs.url(filename)
            full_video_path = fs.path(filename)

            # Procesar el video y realizar la detección
            resultado, detalles = procesar_video(full_video_path)

            # Guardar el incidente en la base de datos
            Incident.objects.create(
                incident_type='VIOLENCE' if detalles['nivel_violencia'] >= 0.9 else 'NON_VIOLENCE',
                location=video_name,
                description=f"Confianza: {detalles['confianza']:.2f} - {video_description}",
                source='UPLOAD'
            )

            return render(request, 'modulos/deteccion_tiemporeal_o_Importacion/video_importado.html', {
                'resultado': resultado,
                'detalles': detalles,
                'video_path': video_path,
                'video_name': video_name,
                'video_description': video_description,
                'form': form
            })

    else:
        form = VideoUploadForm()

    return render(request, 'modulos/deteccion_tiemporeal_o_Importacion/video_importado.html', {'form': form})

def procesar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_deque = deque(maxlen=SEQ_LENGTH)
    predicciones = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar el frame
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
        frames_deque.append(frame_rgb)

        if len(frames_deque) == SEQ_LENGTH:
            frames_array = np.array(list(frames_deque)).reshape(1, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
            prediccion = modelo.predict(frames_array)[0][0]
            predicciones.append(prediccion)

    cap.release()
    nivel_violencia = np.median(predicciones)
    resultado = "Violento" if nivel_violencia >= 0.9 else "No violento"
    detalles = {
        'nivel_violencia': nivel_violencia,
        'confianza': np.max(predicciones)
    }
    return resultado, detalles


def enviar_correo(prediccion_suavizada, buffer, image_content, request):
    if request.user.is_authenticated:
        subject = 'Alerta de Comportamiento Violento'
        message = f"Se ha detectado un comportamiento violento con una confianza de {prediccion_suavizada:.2f}.\n\nUbicación: Cámara en tiempo real"
        email = EmailMessage(
            subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [request.user.email],
        )
        from io import BytesIO
        image_buffer = BytesIO(buffer)
        email.attach(image_content.name, image_buffer.getvalue(), 'image/jpeg')
        email.send()


@gzip.gzip_page
def video_feed(request):
    global transmitir
    transmitir = True
    return StreamingHttpResponse(generar_frames(request), content_type='multipart/x-mixed-replace; boundary=frame')

def stop_feed(request):
    global transmitir
    transmitir = False
    return redirect('camera_view')



@login_required
def incident_list(request):
    incidents_realtime = Incident.objects.filter(source='REAL_TIME').order_by('-timestamp')
    incidents_upload = Incident.objects.filter(source='UPLOAD').order_by('-timestamp')
    return render(request, 'modulos/incident_list.html', {
        'incidents_realtime': incidents_realtime,
        'incidents_upload': incidents_upload
    })

@permission_required('surveillance.delete_incident')
def delete_incident(request, id):
    incident = get_object_or_404(Incident, id=id)
    if request.method == 'POST':
        incident.delete()
        return redirect('incident_list')
    
@permission_required('surveillance.view_incidentreport')   
def report_dashboard(request):
    reports = IncidentReport.objects.all()
    df = pd.DataFrame(list(reports.values('detected_time', 'severity')))
    fig = px.histogram(df, x='detected_time', color='severity')
    chart = fig.to_html()

    return render(request, 'report_dashboard.html', {'chart': chart})


def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()  # Guarda el usuario si el formulario es válido
            messages.success(request, "Registro exitoso. Ahora puedes iniciar sesión.")
            return redirect('login')
        else:
            messages.error(request, "Corrige los errores en el formulario.")
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, "Usuario o contraseña incorrectos.")
    else:
        form = CustomAuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('home')
