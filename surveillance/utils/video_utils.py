import cv2

def iniciar_camara():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la cámara")
    return cap

def leer_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame, (320, 240))  # Ejemplo de redimensionamiento
    return frame
