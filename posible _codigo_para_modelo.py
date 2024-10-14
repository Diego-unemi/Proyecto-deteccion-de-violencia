import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Cargar datos
df = pd.read_csv('datos.csv')

# Ejemplo de preprocesamiento: Extraer características y etiquetas
def extract_features(frame):
    # Aquí debes agregar tu lógica de extracción de características
    # Por ejemplo, puedes calcular el área del contorno y otros parámetros
    return [cv2.contourArea(frame)]

features = []
labels = []

for index, row in df.iterrows():
    video_path = row['file']
    label = row['type']
    
    # Leer el video
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extraer características del cuadro
        features.append(extract_features(frame))
        labels.append(label)
    
    cap.release()

X = np.array(features)
y = np.array(labels)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guardar modelo
import joblib
joblib.dump(model, 'modelo_agresion.pkl')
