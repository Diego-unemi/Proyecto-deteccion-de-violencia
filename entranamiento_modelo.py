import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ruta al archivo CSV
file_path = 'datos.csv'

# Cargar el dataset
try:
    df = pd.read_csv(file_path)
    print(df.head())
except FileNotFoundError:
    print(f'Error: El archivo {file_path} no se encuentra.')
    exit()

# Verificar si la columna objetivo existe
target_column = 'type'
if target_column not in df.columns:
    raise ValueError(f"La columna '{target_column}' no está en el DataFrame.")

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Redimensionar y convertir a escala de grises (como en tu código de detección)
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplanar el frame y añadirlo a las características
        features.extend(frame.flatten())
        
        # Si tenemos suficientes frames, detenemos la extracción
        if len(features) >= 4096:
            break
    
    cap.release()
    
    # Asegurar que tenemos exactamente 4096 características
    features = features[:4096]
    if len(features) < 4096:
        features.extend([0] * (4096 - len(features)))
    
    return np.array(features)

X = []
y = []

for index, row in df.iterrows():
    video_path = row['file']
    label = row[target_column]
    
    if not os.path.exists(video_path):
        print(f"Error: El video {video_path} no existe.")
        continue
    
    features = extract_features(video_path)
    
    if features.size == 0:
        print(f"Error: No se extrajeron características de {video_path}.")
        continue
    
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Guardar el modelo y el scaler
joblib.dump(model, 'modelo_violencia.pkl')
joblib.dump(scaler, 'scaler_violencia.pkl')
print('Modelo y scaler guardados.')