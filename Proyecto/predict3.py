# === predict_balanza_opcional.py ===
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

# === CONFIGURACION ===
ARCHIVO_EXCEL = "files/data_predict/balanza-20241231.xlsx"
HOJA = 0
FILA_ENCABEZADO = 3
MAX_LEN = 20

# === FUNCIONES ===
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ 0-9]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

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

# === CARGAR MODELO Y OBJETOS ===
model = load_model("modelo_mejorado_glove.h5", custom_objects={"focal_loss_fixed": focal_loss()})
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === LEER ARCHIVO DE ENTRADA ===
df = pd.read_excel(ARCHIVO_EXCEL, sheet_name=HOJA, header=FILA_ENCABEZADO)
df.columns = df.columns.str.strip()

if not {"No CUENTA", "CUENTA"}.issubset(df.columns):
    raise ValueError("❌ El archivo debe contener las columnas 'No CUENTA' y 'CUENTA'.")

# PREGUNTAR AL USUARIO SI QUIERE PRESERVAR LOS ENCABEZADOS
opcion = input("¿Deseas conservar las filas sin guion (encabezados)? [s/n]: ").strip().lower()
conservar_encabezados = opcion == 's'

# IDENTIFICAR FILAS DESGLOSADAS
es_desglosada = df["No CUENTA"].astype(str).str.contains("-")

# SEPARAR FILAS
df_desglosadas = df[es_desglosada].copy()
df_encabezados = df[~es_desglosada].copy()

# PREPROCESAR TEXTO PARA PREDICCION
df_desglosadas["No CUENTA"] = df_desglosadas["No CUENTA"].astype(str).str.strip()
df_desglosadas["CUENTA"] = df_desglosadas["CUENTA"].astype(str).str.strip().apply(limpiar_texto)
df_desglosadas["InputText"] = "CUENTA " + df_desglosadas["No CUENTA"] + " NOMBRE " + df_desglosadas["CUENTA"]

X_seq = tokenizer.texts_to_sequences(df_desglosadas["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# PREDICCION
y_pred = model.predict(X).argmax(axis=1)
clases_predichas = [label_encoder.classes_[i] for i in y_pred]
df_desglosadas["Clasificacion_Predicha"] = clases_predichas

# UNIR RESULTADO CON OTRAS FILAS
if conservar_encabezados:
    df_resultado = pd.concat([df_desglosadas, df_encabezados], ignore_index=True).sort_index()
else:
    df_resultado = df_desglosadas

# GUARDAR RESULTADO
salida = ARCHIVO_EXCEL.replace(".xlsx", "_clasificado.xlsx")
df_resultado.to_excel(salida, index=False)
print(f"✅ Archivo generado con predicciones: {salida}")
