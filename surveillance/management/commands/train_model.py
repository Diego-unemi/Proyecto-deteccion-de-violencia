from django.core.management.base import BaseCommand
import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm
from django.conf import settings
from sklearn.model_selection import train_test_split

# Configuración
IMG_SIZE = 64
SEQ_LENGTH = 15
VideoDataDir = os.path.join(settings.BASE_DIR, 'Violence Dataset')  # Ajusta la ruta a tu dataset

class Command(BaseCommand):
    help = 'Entrena el modelo y lo convierte a TensorFlow Lite'

    def handle(self, *args, **kwargs):
        model_dir = settings.MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)  # Crear el directorio de modelos si no existe

        # Función para procesar un video en secuencias de frames
        def video_to_frame_sequence(video_path):
            vidcap = cv2.VideoCapture(video_path)
            frames = []
            count = 0
            while vidcap.isOpened():
                success, frame = vidcap.read()
                if not success:
                    break
                if count % 7 == 0:  # Tomar un frame cada 7 para no sobrecargar la secuencia
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                    frames.append(frame)
                count += 1
            vidcap.release()
            return [frames[i:i + SEQ_LENGTH] for i in range(0, len(frames) - SEQ_LENGTH + 1, SEQ_LENGTH)]

        # Preparar dataset
        def preparar_dataset(video_dir, class_label):
            data, labels = [], []
            for video_file in tqdm(os.listdir(video_dir)):
                video_path = os.path.join(video_dir, video_file)
                sequences = video_to_frame_sequence(video_path)
                for sequence in sequences:
                    data.append(sequence)
                    labels.append(class_label)
            return np.array(data), np.array(labels)

        # Cargar datos
        violence_dir = os.path.join(VideoDataDir, 'Violence')
        non_violence_dir = os.path.join(VideoDataDir, 'NonViolence')
        x_violence, y_violence = preparar_dataset(violence_dir, class_label=1)
        x_non_violence, y_non_violence = preparar_dataset(non_violence_dir, class_label=0)

        X = np.concatenate((x_violence, x_non_violence), axis=0)
        y = np.concatenate((y_violence, y_non_violence), axis=0)
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=73)

        # Crear el modelo CNN-LSTM
        def crear_modelo_cnn_lstm():
            input_tensor = tf.keras.layers.Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
            base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
            base_model.trainable = False
            x = tf.keras.layers.TimeDistributed(base_model)(input_tensor)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
            x = tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
            model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
            return model

        # Entrenar el modelo
        model = crear_modelo_cnn_lstm()
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', mode='min', restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(patience=2, mode='min')
        model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data=(x_val, y_val), callbacks=[early_stopping, lr_scheduler])

        # Guardar el modelo Keras
        model_path = os.path.join(model_dir, "modelo_cnn_lstm.h5")
        model.save(model_path)
        self.stdout.write(self.style.SUCCESS(f"Modelo guardado en {model_path}"))

        # Convertir el modelo a TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,     # Operaciones nativas de TFLite
            tf.lite.OpsSet.SELECT_TF_OPS         # Operaciones adicionales de TensorFlow
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.experimental_enable_resource_variables = True  # Permite manejar variables de recursos en TFLite
        # Convertir el modelo
        tflite_model = converter.convert()

        # Guardar el modelo TensorFlow Lite
        tflite_model_path = os.path.join(model_dir, "modelo_cnn_lstm.tflite")
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        self.stdout.write(self.style.SUCCESS(f"Modelo TensorFlow Lite guardado en {tflite_model_path}"))
