import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# Configuración
IMG_SIZE = 64
SEQ_LENGTH = 15  # Longitud de la secuencia de frames por video
VideoDataDir = 'D:/entrenamiento/Violence Dataset'

# Función para procesar un video en secuencias de frames
def video_to_frame_sequence(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            break
        if count % 7 == 0:  # Tomar un frame cada 7 para no cargar la secuencia
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)
        count += 1

    vidcap.release()

    # Si hay suficientes frames, dividir en secuencias de longitud SEQ_LENGTH
    sequences = []
    for i in range(0, len(frames) - SEQ_LENGTH + 1, SEQ_LENGTH):
        sequence = frames[i:i + SEQ_LENGTH]
        sequences.append(sequence)
    
    return sequences

# Preparar el dataset completo en secuencias de frames
def preparar_dataset(video_dir, class_label):
    data = []
    labels = []

    for video_file in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_file)
        sequences = video_to_frame_sequence(video_path)
        
        for sequence in sequences:
            data.append(sequence)
            labels.append(class_label)

    return np.array(data), np.array(labels)

# Directorios de videos de violencia y no violencia
violence_dir = os.path.join(VideoDataDir, 'Violence')
non_violence_dir = os.path.join(VideoDataDir, 'NonViolence')

# Cargar videos y convertirlos en secuencias
x_violence, y_violence = preparar_dataset(violence_dir, class_label=1)
x_non_violence, y_non_violence = preparar_dataset(non_violence_dir, class_label=0)

# Unir datos y dividir en entrenamiento y prueba
X = np.concatenate((x_violence, x_non_violence), axis=0)
y = np.concatenate((y_violence, y_non_violence), axis=0)

# Dividir en entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=73)

print("Forma de x_train:", x_train.shape)
print("Forma de x_val:", x_val.shape)

# Cargar modelo CNN + LSTM
def crear_modelo_cnn_lstm():
    input_tensor = Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))

    # MobileNetV2 para extraer características de cada frame
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Congelar las capas de MobileNetV2
    
    # Aplicar MobileNetV2 a cada frame individualmente en la secuencia
    x = TimeDistributed(base_model)(input_tensor)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # LSTM para capturar relaciones temporales
    x = LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(x)

    # Capas densas para la clasificación final
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    
    return model

# Crear el modelo y mostrar el resumen
modelo_cnn_lstm = crear_modelo_cnn_lstm()
modelo_cnn_lstm.summary()

# Callbacks para el entrenamiento
patience = 3
start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005
epochs = 30
batch_size = 4

def lrfn(epoch):
    rampup_epochs = 5
    sustain_epochs = 0
    exp_decay = .8
    if epoch < rampup_epochs:
        return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.999:
            print("\nLímite de precisión alcanzado, deteniendo entrenamiento")
            self.model.stop_training = True

end_callback = myCallback()
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=False)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, min_delta=.00075)
lr_plat = tf.keras.callbacks.ReduceLROnPlateau(patience=2, mode='min')

# Entrenar el modelo
history = modelo_cnn_lstm.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                              callbacks=[end_callback, lr_callback, early_stopping, lr_plat])

# Guardar el modelo
modelo_cnn_lstm.save("modelo_cnn_lstm.h5")
print("Modelo guardado como modelo_cnn_lstm.h5")

