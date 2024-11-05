import os
from collections import deque
from datetime import datetime
import threading

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, permission_required
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.core.mail import send_mail, EmailMessage
from django.core.paginator import Paginator
from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators import gzip

from .forms import (
    ContactForm,
    CustomUserCreationForm,
    CustomAuthenticationForm,
    VideoUploadForm,
)
from .models import Incident
from .services.deteccion_servicio import generar_frames
import mimetypes


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
            message = form.cleaned_data['message']
            user_email = request.user.email
            user_name = request.user.get_full_name() or request.user.username

            # Construir el cuerpo del correo
            email_message = f"Petición de contacto de: {user_name} ({user_email})\n\n{message}"

            # Enviar correo
            send_mail(
                "Solicitud de Contacto",
                email_message,
                settings.DEFAULT_FROM_EMAIL,
                [settings.DEFAULT_FROM_EMAIL],
                fail_silently=False,
            )

            messages.success(request, "Tu solicitud ha sido enviada con éxito. Nos pondremos en contacto contigo pronto.")
            return redirect('home')
    else:
        form = ContactForm()

    return render(request, 'about.html', {'form': form})




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


def importar_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)

        # Validación del formulario y del archivo subido
        if form.is_valid():
            file = form.cleaned_data['video_file']
            video_name = form.cleaned_data.get('video_name', 'Video sin nombre')
            video_description = form.cleaned_data.get('video_description', '')

            # Verificar si el archivo subido es de tipo video
            mime_type, _ = mimetypes.guess_type(file.name)
            if not mime_type or not mime_type.startswith('video'):
                messages.error(request, "El archivo seleccionado no es un video. Por favor, suba un archivo de video válido.")
                return redirect('importar_video')

            # Guardar el archivo en el servidor
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            video_path = fs.url(filename)

            # Procesa el video (llama a la función de análisis)
            resultado, detalles = procesar_video(fs.path(filename))

            # Guarda el incidente en la base de datos
            Incident.objects.create(
                incident_type='VIOLENCE' if detalles['nivel_violencia'] >= 0.9 else 'NON_VIOLENCE',
                location=video_name,
                description=f"Confianza: {detalles['confianza']:.2f} - {video_description}",
                source='UPLOAD'
            )

            # Mensajes de éxito o advertencia
            if resultado == "Violento":
                messages.warning(request, "El video contiene contenido violento.")
            else:
                messages.success(request, "El video fue analizado exitosamente y no contiene contenido violento.")

            # Renderiza los resultados del análisis
            return render(request, 'modulos/deteccion_tiemporeal_o_Importacion/video_importado.html', {
                'resultado': resultado,
                'detalles': detalles,
                'video_path': video_path,
                'video_name': video_name,
                'video_description': video_description,
                'form': form
            })
        else:
            messages.error(request, "Hubo un error en el formulario. Verifique los datos e intente nuevamente.")
    else:
        form = VideoUploadForm()

    return render(request, 'nombre_de_tu_template.html', {'form': form})


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
    # Obtener incidentes por cada fuente y ordenarlos por fecha
    incidents_realtime = Incident.objects.filter(source='REAL_TIME').order_by('-timestamp')
    incidents_upload = Incident.objects.filter(source='UPLOAD').order_by('-timestamp')
    
    # Crear paginadores para cada tipo de incidente con 5 elementos por página
    realtime_paginator = Paginator(incidents_realtime, 5)
    upload_paginator = Paginator(incidents_upload, 5)
    
    # Obtener el número de página de los parámetros de la URL
    realtime_page_number = request.GET.get('realtime_page', 1)
    upload_page_number = request.GET.get('upload_page', 1)
    
    # Obtener la página específica del paginador
    incidents_realtime_page = realtime_paginator.get_page(realtime_page_number)
    incidents_upload_page = upload_paginator.get_page(upload_page_number)

    # Pasar las páginas a la plantilla
    return render(request, 'modulos/incident_list.html', {
        'incidents_realtime': incidents_realtime_page,
        'incidents_upload': incidents_upload_page,
    })


@permission_required('surveillance.delete_incident')
def delete_incident(request, id):
    incident = get_object_or_404(Incident, id=id)
    if request.method == 'POST':
        incident.delete()
        return redirect('incident_list')
    

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

from django.http import JsonResponse
from django.template.loader import render_to_string
from .models import Incident  # Ajusta según corresponda

def incident_search_ajax(request):
    # Obtener parámetros de búsqueda
    incident_type = request.GET.get('incident_type', '')
    date = request.GET.get('date', '')

    # Filtrar incidentes en tiempo real
    incidents_realtime = Incident.objects.filter(source='REAL_TIME')
    if incident_type:
        incidents_realtime = incidents_realtime.filter(incident_type=incident_type)
    if date:
        incidents_realtime = incidents_realtime.filter(timestamp__date=date)

    # Filtrar incidentes subidos
    incidents_upload = Incident.objects.filter(source='UPLOAD')
    if incident_type:
        incidents_upload = incidents_upload.filter(incident_type=incident_type)
    if date:
        incidents_upload = incidents_upload.filter(timestamp__date=date)

    # Renderizar tablas en HTML
    realtime_table_html = render_to_string('partials/realtime_table.html', {'incidents_realtime': incidents_realtime})
    upload_table_html = render_to_string('partials/upload_table.html', {'incidents_upload': incidents_upload})

    # Devolver JSON con HTML renderizado para actualizar las tablas en la página
    return JsonResponse({
        'realtime_table': realtime_table_html,
        'upload_table': upload_table_html,
    })

