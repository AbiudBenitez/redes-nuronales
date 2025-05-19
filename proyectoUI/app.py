from flask import Flask, request, send_file
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

# Configuración
MAX_LEN = 20

# Función de limpieza
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# Cargar modelo y utilidades
print("🔧 Cargando modelo...")
model = load_model("modelo_multisalida_glove.h5", compile=False)
print("✅ Modelo cargado.")

print("📦 Cargando tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("✅ Tokenizer cargado.")

print("🎯 Cargando label_encoder...")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
print("✅ Label encoder cargado.")

# Función de predicción
def procesar_y_predecir(df):
    df["Nombre de Cuenta"] = df["Nombre de Cuenta"].apply(limpiar_texto)
    df["InputText"] = "CUENTA " + df["Número de cuenta"].astype(str) + " NOMBRE " + df["Nombre de Cuenta"]
    X_seq = tokenizer.texts_to_sequences(df["InputText"])
    X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

    predicciones = model.predict(X)
    y_clasificacion = predicciones[0].argmax(axis=1)
    y_prpnr = (predicciones[1] > 0.5).astype(int).flatten()

    df["Clasificación_Predicha"] = label_encoder.inverse_transform(y_clasificacion)
    df["PR_PNR_Predicho"] = np.where(y_prpnr == 1, "PR", "PNR")
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["archivo"]
        if file:
            try:
                df = pd.read_excel(file, header=3)
                df.columns = df.columns.str.strip()
                df = df.rename(columns={"No CUENTA": "Número de cuenta", "CUENTA": "Nombre de Cuenta"})
                df = df[df["Número de cuenta"].astype(str).str.contains("-")].copy()
                df_resultado = procesar_y_predecir(df)
                output_path = "resultado.xlsx"
                df_resultado.to_excel(output_path, index=False)
                return send_file(output_path, as_attachment=True)
            except Exception as e:
                return f"<h3>Error al procesar el archivo: {e}</h3>"

    return """
<!DOCTYPE html>
<html>
<head>
    <title>Clasificación Contable + PR/PNR</title>
</head>
<body>
    <h2>Clasificador Contable</h2>
    <form method="POST" enctype="multipart/form-data">
        <label>Sube tu archivo Excel (.xlsx):</label><br>
        <input type="file" name="archivo" accept=".xlsx" required><br><br>
        <input type="submit" value="Ejecutar predicción">
    </form>
</body>
</html>
"""

if __name__ == "__main__":
    print("🚀 Iniciando servidor Flask en http://localhost:5000 ...")
    app.run(debug=True)
