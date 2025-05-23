import pandas as pd
import numpy as np
import re
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, dtype='int32')
        y_true = K.one_hot(y_true, num_classes=K.shape(y_pred)[-1])
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

ARCHIVO_ENTRADA = "files/data_predict/balanza-20241231.xlsx"
MAX_LEN = 20

df = pd.read_excel(ARCHIVO_ENTRADA, header=3)
df.columns = df.columns.str.strip()

if "No CUENTA" not in df.columns or "CUENTA" not in df.columns:
    raise ValueError("❌ El archivo debe tener columnas 'No CUENTA' y 'CUENTA'.")

df = df.rename(columns={"No CUENTA": "Número de cuenta", "CUENTA": "Nombre de Cuenta"})
df["__indice_original__"] = df.index

tiene_guion = df["Número de cuenta"].astype(str).str.contains("-")
df_desglosadas = df[tiene_guion].copy()
df_encabezados = df[~tiene_guion].copy()

opcion = input("¿Deseas conservar los encabezados (cuentas sin guión) en la salida? (s/n): ").strip().lower()
conservar_encabezados = opcion == "s"

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

df_desglosadas["Nombre de Cuenta"] = df_desglosadas["Nombre de Cuenta"].apply(limpiar_texto)
df_desglosadas["InputText"] = (
    "CUENTA " + df_desglosadas["Número de cuenta"].astype(str) +
    " NOMBRE " + df_desglosadas["Nombre de Cuenta"]
)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = load_model("modelo_multisalida_glove.h5", custom_objects={"focal_loss_fixed": focal_loss()})

X_seq = tokenizer.texts_to_sequences(df_desglosadas["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

predicciones = model.predict(X)
y_clasificacion = predicciones[0].argmax(axis=1)
y_prpnr = (predicciones[1] > 0.5).astype(int).flatten()

df_desglosadas["Clasificación_Predicha"] = label_encoder.inverse_transform(y_clasificacion)
df_desglosadas["PR_PNR_Predicho"] = np.where(y_prpnr == 1, "PR", "PNR")
df_desglosadas["Predicho"] = "Sí"

if conservar_encabezados:
    df_encabezados["Clasificación_Predicha"] = ""
    df_encabezados["PR_PNR_Predicho"] = ""
    df_encabezados["Predicho"] = "No"
    df_resultado = pd.concat([df_desglosadas, df_encabezados], ignore_index=True)
    df_resultado = df_resultado.sort_values("__indice_original__").drop(columns="__indice_original__").reset_index(drop=True)
else:
    df_resultado = df_desglosadas.drop(columns="__indice_original__").reset_index(drop=True)

archivo_salida = ARCHIVO_ENTRADA.replace(".xlsx", "_multisalida_predicho.xlsx")
df_resultado.to_excel(archivo_salida, index=False)
print(f"✅ Archivo generado: {archivo_salida}")
