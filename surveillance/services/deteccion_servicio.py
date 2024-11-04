# services/detection_service.py
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from surveillance.models import Incident


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