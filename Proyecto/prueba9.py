# Reejecutamos el script adaptado tras el reinicio del entorno
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout

# === CONFIGURACIÓN ===
CARPETA_DATOS = "files/dataset"
GLOVE_PATH = "files/glove.6B/glove.6B.100d.txt"
NOMBRE_HOJA = "BZ"
FILA_ENCABEZADO = 17
MAX_LEN = 20
NUM_WORDS = 5000
EMBEDDING_DIM = 100
LABEL_OBJETIVO = "Clasificación"

# === 1. Cargar múltiples archivos Excel ===
archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")]
df_completo = []

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

# === 5. Cargar GloVe ===
embeddings_index = {}
with open(GLOVE_PATH, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# === 6. Modelo con GloVe ===
input_layer = Input(shape=(MAX_LEN,))
embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    trainable=False
)(input_layer)
x = GlobalAveragePooling1D()(embedding_layer)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=30, batch_size=16)

# === 7. Evaluación ===
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

# === 8. Guardar modelo y objetos ===
model.save("modelo_mejorado_glove.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
