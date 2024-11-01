# import cv2
<<<<<<< HEAD
# import numpy as np
# import tensorflow as tf
# from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings

# from django.conf import settings
# from collections import deque
from .forms import ContactForm
from .forms import VideoUploadForm


# # Cargar el modelo entrenado
# modelo = tf.keras.models.load_model("modelo_cnn_lstm.h5")

# # Configuración
# IMG_SIZE = 64
# SEQ_LENGTH = 15  # Mantener la secuencia en 15 frames, ya que el modelo fue entrenado así
# UMBRAL = 0.85
# frames_deque = deque(maxlen=SEQ_LENGTH)  # Cola de frames
# predictions_deque = deque(maxlen=5)  # Cola para suavizar predicciones

# def camera_view(request):
#     return render(request, "camera.html")

# def video_feed():
#     cap = cv2.VideoCapture(0)
#     label, color = "Cargando...", (255, 255, 255)  # Inicialización de la etiqueta y color

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Preprocesar el frame
#         frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#         frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
#         frames_deque.append(frame_rgb)

#         # Realizar predicción cuando haya suficientes frames en la cola
#         if len(frames_deque) == SEQ_LENGTH:
#             frames_array = np.array(list(frames_deque)).reshape(1, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
#             prediccion = modelo.predict(frames_array)[0][0]

#             # Añadir la predicción a la cola de suavizado
#             predictions_deque.append(prediccion)
#             avg_prediccion = np.mean(predictions_deque)

#             # Clasificación basada en el promedio de predicciones
#             if avg_prediccion >= UMBRAL:
#                 label = f"Violencia: {avg_prediccion:.2f}"
#                 color = (0, 0, 255)  # Rojo
#             else:
#                 label = f"No Violencia: {avg_prediccion:.2f}"
#                 color = (0, 255, 0)  # Verde

#         # Mostrar el resultado en el frame
#         cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#         # Codificar el frame para la transmisión en el navegador
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()
#     cv2.destroyAllWindows()


=======
# import time
# import threading
# from django.http import StreamingHttpResponse
# from surveillance.services.deteccion_servicio import DeteccionViolenciaService
# from surveillance.utils.video_utils import iniciar_camara, leer_frame
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from .forms import ContactForm
>>>>>>> 1056968 (update views)

def home_view(request):
    return render(request, 'home.html') 

def about_view(request):
    return render(request, 'about.html') 

def dashboard_view(request):
    return render(request, 'dashboard.html')

def video_view(request):
    return render(request, 'video.html') 

<<<<<<< HEAD
=======
def camera_view(request):
    return render(request, "camera.html")

>>>>>>> 1056968 (update views)
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

<<<<<<< HEAD
# from django.http import StreamingHttpResponse
# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque
# import os
# import imgaug.augmenters as iaa
# # Ruta del modelo TensorFlow Lite
# TFLITE_MODEL_PATH = os.path.join(settings.MODEL_DIR, "modelo_cnn_lstm.tflite")

# # Configuración
# IMG_SIZE = 64
# SEQ_LENGTH = 15
# UMBRAL = 0.3  # Ajusta según tu preferencia

# # Cargar el modelo TFLite
# interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
# interpreter.allocate_tensors()

# # Obtener índices de entrada y salida
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Inicializar la cola de frames
# frames_deque = deque(maxlen=SEQ_LENGTH)

# # Definir secuencia de aumento de datos
# augmenter = iaa.Sequential([
#     iaa.Affine(rotate=(-5, 5)),       # Rotación leve
#     iaa.LinearContrast((0.8, 1.2)),   # Contraste variable
# ])

# def preprocess_frame(frame):
#     frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
#     frame_augmented = augmenter(image=frame_rgb)
#     return frame_augmented

# def video_feed():
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Preprocesar el frame y agregarlo a la secuencia
#         processed_frame = preprocess_frame(frame)
#         frames_deque.append(processed_frame)

#         if len(frames_deque) == SEQ_LENGTH:
#             # Convertir los frames en un array de la forma que espera el modelo
#             frames_array = np.array(list(frames_deque), dtype=np.float32).reshape(1, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
            
#             # Realizar la predicción con el modelo TFLite
#             interpreter.set_tensor(input_details[0]['index'], frames_array)
#             interpreter.invoke()
#             prediccion = interpreter.get_tensor(output_details[0]['index'])[0][0]

#             # Determinar la etiqueta en función del umbral
#             if prediccion >= UMBRAL:
#                 label = f"Violencia: {prediccion:.2f}"
#                 color = (0, 0, 255)  # Rojo para violencia
#             else:
#                 label = f"No Violencia: {prediccion:.2f}"
#                 color = (0, 255, 0)  # Verde para no violencia

#             # Mostrar la etiqueta en el frame
#             cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#         # Codificar el frame para transmisión
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

# def video_feed_view(request):
#     return StreamingHttpResponse(video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')




def camera_view(request):
    return render(request, "camera.html")


# # Configuración del modelo TFLite
# TFLITE_MODEL_PATH = settings.TFLITE_MODEL_PATH  # Ruta de tu modelo TensorFlow Lite
# IMG_SIZE = 64
# SEQ_LENGTH = 15
# UMBRAL = 0.7  # Ajusta este umbral según tus pruebas

# # Cargar el modelo TFLite
# interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# def preprocess_frame(frame):
#     frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
#     return frame_rgb

# def process_video(file_path):
#     cap = cv2.VideoCapture(file_path)
#     frames_deque = deque(maxlen=SEQ_LENGTH)
#     results = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Preprocesar el frame
#         processed_frame = preprocess_frame(frame)
#         frames_deque.append(processed_frame)

#         # Realizar la predicción si tenemos suficientes frames en la secuencia
#         if len(frames_deque) == SEQ_LENGTH:
#             frames_array = np.array(list(frames_deque), dtype=np.float32).reshape(1, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
#             interpreter.set_tensor(input_details[0]['index'], frames_array)
#             interpreter.invoke()
#             prediccion = interpreter.get_tensor(output_details[0]['index'])[0][0]

#             # Almacenar resultado de predicción
#             label = "Violencia" if prediccion >= UMBRAL else "No Violencia"
#             results.append((label, prediccion))

#     cap.release()
#     return results

# def video_prediction_view(request):
#     if request.method == "POST":
#         form = VideoUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             video_file = request.FILES['video']
#             video_path = f"{settings.MEDIA_ROOT}/{video_file.name}"
#             with open(video_path, 'wb+') as f:
#                 for chunk in video_file.chunks():
#                     f.write(chunk)
            
#             # Procesar el video y obtener las predicciones
#             results = process_video(video_path)
#             return render(request, "video_results.html", {"results": results, "video_name": video_file.name})
#     else:
#         form = VideoUploadForm()
    
#     return render(request, "video_upload.html", {"form": form})

####

# app/views.py
=======
# def iniciar_streaming():
#     def procesar_stream():
#         cap = iniciar_camara()
#         while True:
#             frame = leer_frame(cap)
#             prediccion = deteccion_servicio.detectar(frame)
#             setattr(deteccion_servicio, 'ultimo_frame', frame)
#             setattr(deteccion_servicio, 'ultima_prediccion', prediccion)

#     hilo_streaming = threading.Thread(target=procesar_stream, daemon=True)
#     hilo_streaming.start()

# deteccion_servicio = DeteccionViolenciaService(modelo_path="models/modelo_cnn_lstm copy.h5")

# def generar_frames():
#     cap = iniciar_camara()
#     while True:
#         frame = leer_frame(cap)
#         if frame is None:
#             break

#         prediccion = deteccion_servicio.detectar(frame)

#         if prediccion is not None:
#             label = f"Violencia: {prediccion:.2f}" if prediccion >= deteccion_servicio.UMBRAL_INICIAL else f"No Violencia: {prediccion:.2f}"
#             color = (0, 0, 255) if prediccion >= deteccion_servicio.UMBRAL_INICIAL else (0, 255, 0)
#             cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#         time.sleep(0.01)  

# def video_feed_view(request):
#     return StreamingHttpResponse(generar_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# Importaciones necesarias
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
>>>>>>> 1056968 (update views)
from django.http import StreamingHttpResponse
from surveillance.services.deteccion_servicio import DeteccionViolenciaService
from surveillance.utils.video_utils import iniciar_camara, leer_frame
import cv2
import time
import threading

def iniciar_streaming():
    def procesar_stream():
        cap = iniciar_camara()
        while True:
            frame = leer_frame(cap)
            prediccion = deteccion_servicio.detectar(frame)
            # Guardar el frame y predicción para acceder en el generador
            setattr(deteccion_servicio, 'ultimo_frame', frame)
            setattr(deteccion_servicio, 'ultima_prediccion', prediccion)

    hilo_streaming = threading.Thread(target=procesar_stream, daemon=True)
    hilo_streaming.start()

# Instancia del servicio
deteccion_servicio = DeteccionViolenciaService(modelo_path="models/modelo_cnn_lstm copy.h5")

def generar_frames():
<<<<<<< HEAD
    cap = iniciar_camara()
    while True:
        frame = leer_frame(cap)
        if frame is None:
            break

        # Realiza la predicción
        prediccion = deteccion_servicio.detectar(frame)

        # Si hay una predicción, muestra el resultado
        if prediccion is not None:
            label = f"Violencia: {prediccion:.2f}" if prediccion >= deteccion_servicio.UMBRAL_INICIAL else f"No Violencia: {prediccion:.2f}"
            color = (0, 0, 255) if prediccion >= deteccion_servicio.UMBRAL_INICIAL else (0, 255, 0)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Agrega un retraso para evitar sobrecarga de CPU
        time.sleep(0.01)  # Ajusta según rendimiento

def video_feed_view(request):
    return StreamingHttpResponse(generar_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
=======
    global frame_anterior, frames_deque, predicciones
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("No se pudo acceder a la cámara.")

    while True:
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
                label = f"Comportamiento violento: {prediccion_suavizada:.2f}" if prediccion_suavizada >= UMBRAL_INICIAL else f"Comportamiento no violento: {prediccion_suavizada:.2f}"
                color = (0, 0, 255) if prediccion_suavizada >= UMBRAL_INICIAL else (0, 255, 0)

                # Visualización en pantalla
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (10, 60), (int(prediccion_suavizada * 300) + 10, 80), color, -1)
                cv2.putText(frame, f"Confianza: {prediccion_suavizada:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

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
    return StreamingHttpResponse(generar_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# En tu archivo urls.py, agrega la siguiente ruta
# urlpatterns = [
#     path('video_feed/', views.video_feed, name='video_feed'),
# ]
>>>>>>> 1056968 (update views)
