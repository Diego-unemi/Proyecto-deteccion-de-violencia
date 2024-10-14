import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
model_file = 'modelo_violencia.pkl'
model = joblib.load(model_file)

# Función para extraer características de un video (similar a la función anterior)
def extract_features(frame, prev_frame_gray):
    # Convertir el frame actual a escala de grises
    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Histograma de colores
    hist = cv2.calcHist([frame], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # 2. Detección de movimiento
    frame_diff = cv2.absdiff(prev_frame_gray, current_frame_gray)
    motion_score = np.sum(frame_diff)  # Sumar valores de la diferencia
    
    # 3. Detección de bordes
    edges = cv2.Canny(frame, 100, 200)
    edge_count = np.sum(edges)  # Contar los bordes detectados
    
    # Agregar las características extraídas a la lista
    features = hist.tolist()
    features.append(motion_score)
    features.append(edge_count)
    
    return np.array(features), current_frame_gray

# Iniciar la captura de video
cap = cv2.VideoCapture(0)  # Cambia el índice si tienes varias cámaras

# Leer el primer frame para inicializar prev_frame_gray
ret, prev_frame = cap.read()
if not ret:
    print("No se pudo acceder a la cámara.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    features, prev_frame_gray = extract_features(frame, prev_frame_gray)
    
    # Rellenar o truncar características a 4096
    if len(features) < 4096:
        features = np.pad(features, (0, 4096 - len(features)), 'constant')
    else:
        features = features[:4096]
    
    # Hacer una predicción
    prediction = model.predict(features.reshape(1, -1))
    
    # Verificar la predicción
    if prediction[0] == 'aggressive':
        # Dibujar un rectángulo rojo si se detecta agresión
        frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)  # Red
    
    # Convertir el frame a RGB para mostrar con matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Mostrar el frame con matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()
    
    # Salir del loop al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
