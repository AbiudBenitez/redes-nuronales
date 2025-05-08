import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout

# === CONFIGURACIÓN ===
CARPETA_DATOS = "files/dataset"  # Ruta a tu carpeta de archivos
NOMBRE_HOJA = "BZ"
FILA_ENCABEZADO = 17
MAX_LEN = 20
NUM_WORDS = 5000
LABEL_OBJETIVO = "Clasificación"

# === 1. Cargar múltiples archivos Excel ===
archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")]
df_completo = []

print(f"📦 Encontrados {len(archivos)} archivos...")

for archivo in archivos:
    ruta = os.path.join(CARPETA_DATOS, archivo)
    try:
        df = pd.read_excel(ruta, sheet_name=NOMBRE_HOJA, header=FILA_ENCABEZADO)
        df["__archivo_origen__"] = archivo
        df_completo.append(df)
    except Exception as e:
        print(f"⚠️ Error leyendo {archivo}: {e}")

df = pd.concat(df_completo, ignore_index=True)
df.columns = df.columns.str.strip()

# === 2. Limpieza y preparación ===
for col in ["Número de cuenta", "Nombre de Cuenta", LABEL_OBJETIVO]:
    if col not in df.columns:
        raise ValueError(f"❌ Falta la columna requerida: {col}")

df["Número de cuenta"] = df["Número de cuenta"].astype(str).str.strip()
df["Nombre de Cuenta"] = df["Nombre de Cuenta"].astype(str).str.strip()
df[LABEL_OBJETIVO] = df[LABEL_OBJETIVO].astype(str).str.strip()

df.dropna(subset=[LABEL_OBJETIVO], inplace=True)

# Entrada enriquecida con contexto
df["InputText"] = (
    "CUENTA " + df["Número de cuenta"] +
    " NOMBRE " + df["Nombre de Cuenta"] +
    " ORIGEN " + df["__archivo_origen__"]
)

# === 3. Tokenización ===
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["InputText"])
X_seq = tokenizer.texts_to_sequences(df["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# === 4. Etiquetas ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[LABEL_OBJETIVO])
num_classes = len(label_encoder.classes_)

print(f"🔢 Datos: {len(X)} muestras | Clases únicas: {num_classes}")

# === 5. Modelo mejorado ===
input_layer = Input(shape=(MAX_LEN,))
embedding = Embedding(input_dim=NUM_WORDS, output_dim=64)(input_layer)
x = GlobalAveragePooling1D()(embedding)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=30, batch_size=16)

# === 6. Evaluación ===
print("📈 Evaluando rendimiento del modelo...")
y_pred = model.predict(X).argmax(axis=1)
print("\n📊 Reporte de clasificación:")
print(classification_report(y, y_pred, target_names=label_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta verdadera")
plt.title("Matriz de Confusión")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# === 7. Guardar modelo y objetos ===
print("💾 Guardando modelo y componentes...")
model.save("modelo_mejorado.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ ¡Modelo entrenado, evaluado y guardado correctamente!")
