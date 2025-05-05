# entrenar_modelo_sat.py

import pandas as pd
import pdfplumber
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
import os

# === CONFIGURACIONES ===
PDF_PATH = "C:/Users/abiud/Documents/Code/redes-nuronales/Proyecto/codigo_agrupador.pdf"
EXCEL_PATH = (
    "C:/Users/abiud/Documents/Code/redes-nuronales/Proyecto/Archivo Anual 2024.xlsx"
)
EXCEL_HOJA_DATOS = "BZ"  # Cambia este nombre por la hoja que tenga el número de cuenta
EXCEL_HOJA_CLASIFICACION = "Clasificaciones"
NUM_WORDS = 5000
MAX_LEN = 20

# === 1. EXTRAER TABLA DEL PDF DEL SAT ===
print("🔍 Extrayendo tabla del PDF...")
all_rows = []
columnas_detectadas = None

with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if not table:
            continue

        # Extraer encabezado si no se ha establecido
        if columnas_detectadas is None:
            posibles_columnas = table[0]
            if posibles_columnas and all(isinstance(c, str) for c in posibles_columnas):
                columnas_detectadas = posibles_columnas
            else:
                continue

        # Filtrar filas válidas
        for fila in table[1:]:
            if len(fila) == len(columnas_detectadas):
                all_rows.append(fila)

# Crear DataFrame final con columnas reales
df_pdf_raw = pd.DataFrame(all_rows, columns=columnas_detectadas)

# Verifica los nombres para evitar errores
print("📋 Columnas detectadas en el PDF:")
print(df_pdf_raw.columns)

# Buscar columnas que contengan las palabras clave
codigo_col = [col for col in df_pdf_raw.columns if col and "código" in col.lower()][0]
nombre_col = [col for col in df_pdf_raw.columns if col and "nombre" in col.lower()][0]

df_pdf = df_pdf_raw[[codigo_col, nombre_col]].copy()
df_pdf.columns = ["NumeroCuenta", "NombreCuenta"]

# === 2. LEER ARCHIVO EXCEL Y UNIR ===
print("📄 Leyendo archivo Excel...")
df_excel = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_HOJA_DATOS)
df_clasif = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_HOJA_CLASIFICACION)

# === 3. UNIR TABLAS POR NUMERO Y NOMBRE DE CUENTA ===
print("🔗 Uniendo datos...")
df_merge = pd.merge(
    df_excel, df_pdf, left_on="Número de cuenta", right_on="NumeroCuenta", how="left"
)
df_final = pd.merge(
    df_merge, df_clasif, left_on="NombreCuenta", right_on="Nombre Cuenta", how="left"
)

# Verificamos datos válidos
df_final.dropna(
    subset=[
        "Estado de Situación Financiera y Estado de Resultados General",
        "Estado de Flujo de Efectivo Indirecto",
        "Estado de Cambios en el Capital",
        "Ingresos Nominales y Deducciones Autorizadas",
    ],
    inplace=True,
)

# === 4. PREPARAR ENTRADAS Y ETIQUETAS ===
print("✍️ Preparando datos para el modelo...")
df_final["InputText"] = (
    df_final["NumeroCuenta"].astype(str) + " " + df_final["NombreCuenta"]
)

tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df_final["InputText"])
sequences = tokenizer.texts_to_sequences(df_final["InputText"])
X = pad_sequences(sequences, padding="post", maxlen=MAX_LEN)

# === 5. ENCODING DE ETIQUETAS ===
etiquetas = [
    "Estado de Situación Financiera y Estado de Resultados General",
    "Estado de Flujo de Efectivo Indirecto",
    "Estado de Cambios en el Capital",
    "Ingresos Nominales y Deducciones Autorizadas",
]

label_encoders = {}
y = []

for columna in etiquetas:
    le = LabelEncoder()
    df_final[columna] = le.fit_transform(df_final[columna].astype(str))
    label_encoders[columna] = le
    y.append(df_final[columna].values)

# === 6. CONSTRUIR Y ENTRENAR MODELO ===
print("🤖 Entrenando modelo con Keras...")
input_layer = Input(shape=(MAX_LEN,))
embedding = Embedding(input_dim=NUM_WORDS, output_dim=64, input_length=MAX_LEN)(
    input_layer
)
pooling = GlobalAveragePooling1D()(embedding)
dense = Dense(64, activation="relu")(pooling)

outputs = []
for columna in etiquetas:
    n_clases = df_final[columna].nunique()
    outputs.append(Dense(n_clases, activation="softmax", name=columna)(dense))

model = Model(inputs=input_layer, outputs=outputs)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(X, y, epochs=10, batch_size=32)

# === 7. GUARDAR MODELO Y OBJETOS ===
print("💾 Guardando modelo y componentes...")
model.save("modelo_sat_multietiqueta.h5")

import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ ¡Listo! Modelo entrenado y guardado correctamente.")
