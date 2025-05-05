# entrenar_modelo_bz.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
import pickle

# === CONFIGURACIÓN ===
EXCEL_PATH = "files/Archivo Anual 2024.xlsx"
EXCEL_HOJA_BZ = "BZ"
MAX_LEN = 20
NUM_WORDS = 5000
LABEL_OBJETIVO = "Clasificación"

# === 1. LEER HOJA BZ ===
print("📄 Leyendo hoja BZ...")
df = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_HOJA_BZ, header=17)
df.columns = df.columns.str.strip()

# Validación de columnas requeridas
required_cols = ["Número de cuenta", "Nombre de Cuenta", "Clasificación"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Falta la columna requerida: {col}")

# Limpiar datos
df["Número de cuenta"] = df["Número de cuenta"].astype(str).str.strip()
df["Nombre de Cuenta"] = df["Nombre de Cuenta"].astype(str).str.strip()
df[LABEL_OBJETIVO] = df[LABEL_OBJETIVO].astype(str).str.strip()

# Eliminar filas sin clasificación
df.dropna(subset=[LABEL_OBJETIVO], inplace=True)

# === 2. CREAR TEXTO DE ENTRADA ===
df["InputText"] = df["Número de cuenta"] + " " + df["Nombre de Cuenta"]

# === 3. TOKENIZACIÓN ===
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["InputText"])
X_seq = tokenizer.texts_to_sequences(df["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# === 4. ETIQUETAS ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[LABEL_OBJETIVO])
num_classes = len(label_encoder.classes_)

print(f"🔢 Registros para entrenamiento: {len(X)} | Clases: {num_classes}")

# === 5. MODELO ===
print("🤖 Entrenando modelo...")
input_layer = Input(shape=(MAX_LEN,))
embedding = Embedding(input_dim=NUM_WORDS, output_dim=64)(input_layer)
pooling = GlobalAveragePooling1D()(embedding)
dense = Dense(64, activation="relu")(pooling)
output = Dense(num_classes, activation="softmax")(dense)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10, batch_size=32)

# === 6. GUARDAR MODELO Y OBJETOS ===
print("💾 Guardando modelo y componentes...")
