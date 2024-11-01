import cv2
import numpy as np
import tensorflow as tf
from collections import deque

class DeteccionViolenciaService:
    def __init__(self, modelo_path="ruta/del/modelo.h5", img_size=64, seq_length=15, umbral_inicial=0.5, umbral_movimiento=20000):
        self.modelo = tf.keras.models.load_model(modelo_path)
        self.IMG_SIZE = img_size
        self.SEQ_LENGTH = seq_length
        self.UMBRAL_INICIAL = umbral_inicial
        self.UMBRAL_MOVIMIENTO = umbral_movimiento
        self.frames_deque = deque(maxlen=seq_length)
        self.predicciones = deque(maxlen=5)
        self.frame_anterior = None

    def preprocesar_frame(self, frame):
        h, w, _ = frame.shape
        crop_size = int(min(h, w) * 0.8)
        y1, x1 = (h - crop_size) // 2, (w - crop_size) // 2
        frame_cropped = frame[y1:y1 + crop_size, x1:x1 + crop_size]
        frame_resized = cv2.resize(frame_cropped, (self.IMG_SIZE, self.IMG_SIZE))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
        return frame_rgb

    def detectar(self, frame):
        frame_rgb = self.preprocesar_frame(frame)
        self.frames_deque.append(frame_rgb)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        movimiento = np.sum(cv2.absdiff(self.frame_anterior, gray_frame)) if self.frame_anterior is not None else 0

        if movimiento > self.UMBRAL_MOVIMIENTO and len(self.frames_deque) == self.SEQ_LENGTH:
            frames_array = np.array(list(self.frames_deque)).reshape(1, self.SEQ_LENGTH, self.IMG_SIZE, self.IMG_SIZE, 3)
            prediccion = self.modelo.predict(frames_array)[0][0]
            self.predicciones.append(prediccion)
            self.frame_anterior = gray_frame
<<<<<<< HEAD
            return np.mean(self.predicciones)  # Suavizado
=======
            return np.mean(self.predicciones) 
>>>>>>> 1056968 (update views)
        self.frame_anterior = gray_frame
        return None