import cv2
from django.http import StreamingHttpResponse, HttpResponse
from django.views.decorators import gzip
from django.shortcuts import render, redirect
import joblib
import numpy as np
import threading
from .forms import ContactForm
from django.core.mail import send_mail
from django.conf import settings

# Carga del modelo y sustractor de fondo al nivel del módulo
model = joblib.load('modelo_violencia.pkl')
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

class VideoCamera(object):
    def __init__(self, model, bg_subtractor):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise Exception("No se puede abrir la cámara")
        self.model = model
        self.bg_subtractor = bg_subtractor
        self.frame_count = 0
        self.lock = threading.Lock()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        with self.lock:
            ret, frame = self.video.read()
            if not ret:
                return None

            # Aplica la sustracción de fondo
            fg_mask = self.bg_subtractor.apply(frame)

            # Encuentra los contornos de las áreas en movimiento
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Dibuja los contornos en el cuadro original
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 255, 0), 2)

                    # Realizar la predicción cada 5 cuadros
                    if self.frame_count % 5 == 0:
                        roi = frame[y:y+h, x:x+w]
                        roi = cv2.resize(roi, (64, 64))
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        roi = roi.reshape(1, -1)

                        # Predicción de agresión
                        pred = self.model.predict(roi)

                        if pred == 1:
                            cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 0, 255), 2)

            self.frame_count += 1

            # Convierte el cuadro a formato JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                return jpeg.tobytes()
            else:
                return None

# Instancia única de VideoCamera
camera = VideoCamera(model, bg_subtractor)

def home_view(request):
    return render(request, 'home.html') 

def about_view(request):
    return render(request, 'about.html') 

def dashboard_view(request):
    return render(request, 'dashboard.html') 

def camera_view(request):
    return render(request, 'camera.html')

def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Procesar la información del formulario
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            message = form.cleaned_data['message']

            # Enviar un correo electrónico
            send_mail(
                f'Mensaje de {name}',
                message,
                email,
                [settings.DEFAULT_FROM_EMAIL],
                fail_silently=False,
            )
            return redirect('home')  # Redirige a la página de inicio después de enviar
    else:
        form = ContactForm()

    return render(request, 'contact.html', {'form': form})

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def video_feed(request):
    try:
        return StreamingHttpResponse(gen(camera), content_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:  # Maneja errores específicos
        print(f"Error en video_feed: {str(e)}")
        return HttpResponse(status=500)

# import cv2
# from django.http import StreamingHttpResponse, HttpResponseServerError
# from django.views.decorators import gzip
# import joblib
# import numpy as np
# import threading
# import time

# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         self.is_running = True
#         self.lock = threading.Lock()
#         self.frame = None
#         self.model = joblib.load('modelo_violencia.pkl')
#         self.scaler = joblib.load('scaler_violencia.pkl')
#         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
#         threading.Thread(target=self._capture_loop, daemon=True).start()

#     def __del__(self):
#         self.is_running = False
#         if self.video.isOpened():
#             self.video.release()

#     def _capture_loop(self):
#         while self.is_running:
#             ret, frame = self.video.read()
#             if not ret:
#                 time.sleep(0.1)
#                 continue
            
#             with self.lock:
#                 self.frame = frame

#     def get_frame(self):
#         with self.lock:
#             if self.frame is None:
#                 return None
#             frame = self.frame.copy()

#         fg_mask = self.bg_subtractor.apply(frame)
        
#         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         for contour in contours:
#             if cv2.contourArea(contour) > 500:
#                 (x, y, w, h) = cv2.boundingRect(contour)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
#                 roi = frame[y:y+h, x:x+w]
#                 roi = cv2.resize(roi, (64, 64))
#                 roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#                 roi = roi.reshape(1, -1)
                
#                 try:
#                     roi_scaled = self.scaler.transform(roi)
#                     pred = self.model.predict(roi_scaled)
                    
#                     if pred == 1:  # Asumiendo que 1 representa "violencia"
#                         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#                         cv2.putText(frame, "VIOLENCIA", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
#                 except Exception as e:
#                     print(f"Error en la predicción: {e}")

#         _, jpeg = cv2.imencode('.jpg', frame)
#         return jpeg.tobytes()

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         if frame is not None:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         else:
#             break

# @gzip.gzip_page
# def video_feed(request):
#     try:
#         cam = VideoCamera()
#         return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
#     except Exception as e:
#         return HttpResponseServerError(f"Error: {str(e)}")

# import cv2
# import numpy as np
# import threading
# import time
# import logging
# from django.conf import settings
# import os
# from django.http import StreamingHttpResponse, HttpResponseServerError
# from django.views.decorators import gzip

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# class VideoProcessor(object):
#     def __init__(self, video_path):
#         self.video_path = os.path.join(settings.MEDIA_ROOT, video_path)
#         self.video = cv2.VideoCapture(self.video_path)
#         if not self.video.isOpened():
#             logger.error(f"No se pudo abrir el video: {self.video_path}")
#             raise ValueError(f"No se pudo abrir el video: {self.video_path}")
        
#         self.fps = self.video.get(cv2.CAP_PROP_FPS)
#         self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.duration = self.frame_count / self.fps
        
#         logger.info(f"Video cargado: {self.video_path}")
#         logger.info(f"FPS: {self.fps}, Duración: {self.duration} segundos")
        
#         self.is_running = True
#         self.lock = threading.Lock()
#         self.frame = None
#         self.current_frame = 0
#         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
#         self.last_frame = None
#         self.violence_counter = 0
#         threading.Thread(target=self._process_loop, daemon=True).start()

#     def __del__(self):
#         self.is_running = False
#         if self.video.isOpened():
#             self.video.release()

#     def _process_loop(self):
#         while self.is_running:
#             ret, frame = self.video.read()
#             if not ret:
#                 logger.info("Fin del video alcanzado, reiniciando...")
#                 self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 self.current_frame = 0
#                 continue
            
#             with self.lock:
#                 self.frame = frame
#                 self.current_frame += 1

#             logger.debug(f"Frame {self.current_frame} leído con éxito")

#             # Simular la velocidad real del video
#             time.sleep(1 / self.fps)

#     def get_frame(self):
#         with self.lock:
#             if self.frame is None:
#                 logger.warning("No hay frame disponible")
#                 return None
#             frame = self.frame.copy()

#         # Convertir a escala de grises
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Aplicar sustracción de fondo
#         fg_mask = self.bg_subtractor.apply(gray)
        
#         # Encontrar contornos
#         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Detectar movimiento significativo
#         movement_detected = False
#         for contour in contours:
#             if cv2.contourArea(contour) > 500:  # Ajusta este valor según sea necesario
#                 movement_detected = True
#                 break
        
#         # Simulación simple de detección de violencia basada en cambios rápidos
#         violence_detected = False
#         if self.last_frame is not None:
#             # Calcular la diferencia entre frames
#             frame_diff = cv2.absdiff(gray, self.last_frame)
#             _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            
#             # Si hay muchos píxeles que cambiaron, incrementar el contador de violencia
#             if np.sum(threshold) > 100000:  # Ajusta este valor según sea necesario
#                 self.violence_counter += 1
#             else:
#                 self.violence_counter = max(0, self.violence_counter - 1)
            
#             if self.violence_counter > 5:  # Ajusta este umbral según sea necesario
#                 violence_detected = True
        
#         self.last_frame = gray
        
#         # Dibujar rectángulos alrededor de las áreas de movimiento
#         for contour in contours:
#             if cv2.contourArea(contour) > 500:
#                 (x, y, w, h) = cv2.boundingRect(contour)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         # Crear una barra de estado en la parte inferior
#         status_bar = np.zeros((100, frame.shape[1], 3), dtype=np.uint8)
        
#         # Añadir información sobre el progreso del video y detección
#         progress = self.current_frame / self.frame_count
#         cv2.putText(status_bar, f"Progreso: {progress:.2%}", (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         if movement_detected:
#             cv2.putText(status_bar, "Movimiento detectado", (10, 60), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         if violence_detected:
#             cv2.putText(status_bar, "ALERTA: Posible violencia", (10, 90), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         violence_level = min(self.violence_counter / 10, 1.0)  
#         bar_width = int(violence_level * status_bar.shape[1])
#         cv2.rectangle(status_bar, (0, 70), (bar_width, 80), (0, 0, 255), -1)
        
#         # Combinar el frame con la barra de estado
#         result = np.vstack((frame, status_bar))

#         try:
#             _, jpeg = cv2.imencode('.jpg', result)
#             return jpeg.tobytes()
#         except Exception as e:
#             logger.error(f"Error al codificar el frame: {e}")
#             return None

# def gen(processor):
#     while True:
#         frame = processor.get_frame()
#         if frame is not None:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         else:
#             break

# @gzip.gzip_page
# def video_feed(request):
#     try:
#         video_path = '2.mp4'  
#         processor = VideoProcessor(video_path)
#         return StreamingHttpResponse(gen(processor), content_type="multipart/x-mixed-replace;boundary=frame")
#     except Exception as e:
#         logger.error(f"Error en video_feed: {e}")
#         return HttpResponseServerError(f"Error: {str(e)}")
