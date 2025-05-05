# entrenar_clasificacion_sat.py

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
EXCEL_HOJA_CLASIF = "Clasificaciones"
MAX_LEN = 20
NUM_WORDS = 5000
LABEL_OBJETIVO = "Estado de Situación Financiera y Estado de Resultados General"

# === 1. LEER DATOS ===
print("📄 Leyendo hoja BZ y Clasificaciones...")
df_bz = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_HOJA_BZ, header=17)
df_clasif = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_HOJA_CLASIF, header=1)
df_bz.columns = df_bz.columns.str.strip()
df_clasif.columns = df_clasif.columns.str.strip()
df_clasif.rename(columns={df_clasif.columns[0]: "NombreCuenta"}, inplace=True)
print("🧾 Columnas exactas en df_clasif:")
for col in df_clasif.columns:
    print(f"- '{col}'")
print("🧾 Columnas exactas en df_bz:")
for col in df_bz.columns:
    print(f"- '{col}'")

# === 2. UNIR NOMBRE DE CUENTA CON CLASIFICACIÓN ===
print("🔗 Haciendo merge por NombreCuenta...")
df_bz["NombreCuenta"] = df_bz["Nombre de Cuenta"].astype(str).str.strip()
df_clasif["NombreCuenta"] = df_clasif["NombreCuenta"].astype(str).str.strip()

print("📋 Ejemplo de nombres en df_bz:", df_bz["NombreCuenta"].unique()[:5])
print("📋 Ejemplo de nombres en df_clasif:", df_clasif["NombreCuenta"].unique()[:5])

df = pd.merge(df_bz, df_clasif, on="NombreCuenta", how="left")
print("🔍 Muestras antes del dropna:", df.shape[0])
df.dropna(subset=[LABEL_OBJETIVO], inplace=True)
print("🧹 Muestras después del dropna:", df.shape[0])

print("📊 Columnas en df después del merge:")
print(df.columns.tolist())

print("🔍 Muestras después del merge:")
print(df[["NombreCuenta"]].head())

df_test = pd.merge(df_bz, df_clasif, on="NombreCuenta", how="inner")
print("✅ Coincidencias reales encontradas:", len(df_test))

# === 3. CREAR TEXTO DE ENTRADA ===
print("✍️ Preparando texto de entrada...")
df["NumeroCuenta"] = df["Número de cuenta"].astype(str).str.strip()
df["InputText"] = df["NumeroCuenta"] + " " + df["NombreCuenta"]

# === 4. TOKENIZACIÓN ===
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["InputText"])
X_seq = tokenizer.texts_to_sequences(df["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# === 5. ETIQUETAS ===
label_encoder = LabelEncoder()
df[LABEL_OBJETIVO] = label_encoder.fit_transform(df[LABEL_OBJETIVO].astype(str))
y = df[LABEL_OBJETIVO].values
num_classes = df[LABEL_OBJETIVO].nunique()

print("🔢 Registros totales para entrenamiento:", len(X))

# === 6. MODELO ===
print("🤖 Entrenando modelo...")
input_layer = Input(shape=(MAX_LEN,))
embedding = Embedding(input_dim=NUM_WORDS, output_dim=64)(input_layer)
pooling = GlobalAveragePooling1D()(embedding)
dense = Dense(64, activation="relu")(pooling)
output = Dense(num_classes, activation="softmax")(dense)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10, batch_size=32)

# === 7. GUARDAR MODELO Y OBJETOS ===
print("💾 Guardando modelo y componentes...")
model.save("modelo_sat_clasificacion.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ ¡Entrenamiento y guardado completados!")
