import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("modelo_cnn_lstm.h5")

# Configuración
IMG_SIZE = 64  # Tamaño de los frames según tu entrenamiento
SEQ_LENGTH = 15  # Longitud de la secuencia de frames para la predicción

# Inicializar la cola de frames para crear secuencias
frames_deque = deque(maxlen=SEQ_LENGTH)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar el frame
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
    frames_deque.append(frame_rgb)

    # Solo hacer predicciones cuando haya suficientes frames en la cola
    if len(frames_deque) == SEQ_LENGTH:
        # Convertir los frames en un array de la forma (1, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)
        frames_array = np.array(list(frames_deque)).reshape(1, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)

        # Realizar la predicción
        prediccion = modelo.predict(frames_array)[0][0]

        # Establecer el umbral de clasificación (ajusta según sea necesario)
        umbral = 0.7
        if prediccion >= umbral:
            label = f"Violencia: {prediccion:.2f}"
            color = (0, 0, 255)  # Rojo para violencia
        else:
            label = f"No Violencia: {prediccion:.2f}"
            color = (0, 255, 0)  # Verde para no violencia

        # Mostrar el resultado en el frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Mostrar el frame
    cv2.imshow("Detección de Violencia en Tiempo Real", frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
