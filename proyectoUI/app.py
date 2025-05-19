from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

MAX_LEN = 20

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-záéíóúüñ0-9\\s]", "", texto)
    texto = re.sub(r"\\s+", " ", texto).strip()
    return texto

print("🔧 Cargando modelo...")
model = load_model("modelo_multisalida_glove.h5", compile=False)
print("✅ Modelo cargado.")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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
    error = None
    tabla_html = None

    if request.method == "POST":
        file = request.files.get("archivo")
        if not file:
            error = "No se seleccionó archivo."
        else:
            try:
                df = pd.read_excel(file, header=3)
                df.columns = df.columns.str.strip()

                if "No CUENTA" not in df.columns or "CUENTA" not in df.columns:
                    error = "El archivo debe tener las columnas 'No CUENTA' y 'CUENTA'"
                else:
                    df = df.rename(columns={"No CUENTA": "Número de cuenta", "CUENTA": "Nombre de Cuenta"})
                    df = df[df["Número de cuenta"].astype(str).str.contains("-")].copy()
                    df_resultado = procesar_y_predecir(df)
                    tabla_html = df_resultado.head(20).to_html(classes="preview", index=False)
                    df_resultado.to_excel("resultado.xlsx", index=False)
            except Exception as e:
                error = f"Error al procesar el archivo: {str(e)}"

    return render_template("index.html", error=error, tabla_html=tabla_html)

@app.route("/descargar")
def descargar():
    return send_file("resultado.xlsx", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
