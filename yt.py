import pickle

# Cargar el modelo desde el archivo .pkl
model_file = 'modelo_violencia.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Imprimir los parámetros del modelo
print("Parámetros del modelo:", model.get_params())

# Puedes realizar predicciones o evaluar el modelo de la misma manera que con joblib:
# predictions = model.predict(X_test)
# accuracy = model.score(X_test, y_test)
