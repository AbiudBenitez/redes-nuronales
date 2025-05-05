# entrenar_modelo_sat_tabula.py

import pandas as pd
import tabula
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
import pickle
import os

# === CONFIGURACIONES ===
CODES_PATH = (
    "C:/Users/abiud/Documents/Code/redes-nuronales/Proyecto/codigo_agrupador.xlsx"
)
EXCEL_PATH = (
    "C:/Users/abiud/Documents/Code/redes-nuronales/Proyecto/Archivo Anual 2024.xlsx"
)
EXCEL_HOJA_DATOS = "BZ"  # Cambia por el nombre real de la hoja con "Número de cuenta"
EXCEL_HOJA_CLASIFICACION = "Clasificaciones"
NUM_WORDS = 5000
MAX_LEN = 20

# === 1. LEER CÓDIGO AGRUPADOR DESDE EXCEL ===
print("📦 Leyendo archivo de código agrupador (Excel)...")
df_pdf = pd.read_excel(CODES_PATH)

# Asegúrate de que estas columnas existan y estén bien nombradas:
# ['Código agrupador', 'Nombre de la cuenta y/o subcuenta']

# Renombramos para estandarizar
df_pdf = df_pdf.rename(
    columns={
        "Código agrupador": "NumeroCuenta",
        "Nombre de la cuenta y/o subcuenta": "NombreCuenta",
    }
)

df_pdf = df_pdf[["NumeroCuenta", "NombreCuenta"]].dropna()

print("✅ Código agrupador cargado:")
print(df_pdf.head())

# === 2. LEER ARCHIVO EXCEL ===
print("📄 Leyendo archivo Excel...")
df_excel = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_HOJA_DATOS, header=17)
df_clasif = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_HOJA_CLASIFICACION, header=1)
df_clasif.columns = df_clasif.columns.str.strip()
# df_clasif.rename(columns={df_clasif.columns[1]: "NombreCuenta"}, inplace=True)

print("📑 Columnas de Clasificaciones:")
print(df_clasif.columns.tolist())

print("📊 Columnas del archivo Excel:")
print(df_excel.columns.tolist())

df_excel.columns = df_excel.columns.str.strip()
df_clasif.columns = df_clasif.columns.str.strip()

# === 3. UNIR DATOS ===
print("🔗 Uniendo datos...")

# Normalizar tipo de dato antes del merge
df_excel["Número de cuenta"] = df_excel["Número de cuenta"].astype(str).str.strip()
df_pdf["NumeroCuenta"] = df_pdf["NumeroCuenta"].astype(str).str.strip()

df_merge = pd.merge(
    df_excel, df_pdf, left_on="Número de cuenta", right_on="NumeroCuenta", how="left"
)
df_final = pd.merge(
    df_merge,
    df_clasif,
    left_on="NombreCuenta",
    right_on="Estado de Situación Financiera y Estado de Resultados General",
    how="left",
)

# Eliminar registros con clasificaciones faltantes
df_final.dropna(
    subset=[
        "Estado de Situación Financiera y Estado de Resultados General",
        "Estado de Flujo de Efectivo Indirecto",
        "Estado de Cambios en el Capital",
        "Ingresos Nominales y Deducciones Autorizadas",
    ],
    inplace=True,
)

# === 4. PREPARAR DATOS PARA EL MODELO ===
print("✍️ Preparando texto y etiquetas...")
df_final["InputText"] = (
    df_final["NumeroCuenta"].astype(str) + " " + df_final["NombreCuenta"]
)

# Tokenización del texto
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df_final["InputText"])
X_seq = tokenizer.texts_to_sequences(df_final["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# Codificación de etiquetas
etiquetas = [
    "Estado de Situación Financiera y Estado de Resultados General",
    "Estado de Flujo de Efectivo Indirecto",
    "Estado de Cambios en el Capital",
    "Ingresos Nominales y Deducciones Autorizadas",
]

label_encoders = {}
y = []

for col in etiquetas:
    le = LabelEncoder()
    df_final[col] = le.fit_transform(df_final[col].astype(str))
    label_encoders[col] = le
    y.append(df_final[col].values)

# === 5. CONSTRUIR Y ENTRENAR MODELO CON KERAS ===
print("🤖 Entrenando modelo con Keras...")
input_layer = Input(shape=(MAX_LEN,))
embedding = Embedding(input_dim=NUM_WORDS, output_dim=64, input_length=MAX_LEN)(
    input_layer
)
pooling = GlobalAveragePooling1D()(embedding)
dense = Dense(64, activation="relu")(pooling)

outputs = []
for col in etiquetas:
    num_classes = df_final[col].nunique()
    outputs.append(Dense(num_classes, activation="softmax", name=col)(dense))

model = Model(inputs=input_layer, outputs=outputs)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(X, y, epochs=10, batch_size=32)

# === 6. GUARDAR MODELO Y OBJETOS ===
print("💾 Guardando modelo y componentes...")
model.save("modelo_sat_multietiqueta.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ ¡Modelo entrenado y guardado correctamente!")
