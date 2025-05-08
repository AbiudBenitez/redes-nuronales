import numpy as np
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
EXCEL_PATH = "files/dataset/Archivo Anual 2024.xlsx"
HOJA_BZ = "BZ"
HOJA_CLASIF = "Clasificaciones"
FILA_BZ = 17
FILA_CLASIF = 1
MAX_LEN = 20
NUM_WORDS = 5000
EMBEDDING_DIM = 100
LABEL_OBJETIVO = "Estado de Situación Financiera y Estado de Resultados General"
GLOVE_PATH = "files/glove.6B/glove.6B.100d.txt"  # Ruta local del archivo GloVe

# === 1. Cargar datos ===
print("📄 Leyendo hojas BZ y Clasificaciones...")
df_bz = pd.read_excel(EXCEL_PATH, sheet_name=HOJA_BZ, header=FILA_BZ)
df_clasif = pd.read_excel(EXCEL_PATH, sheet_name=HOJA_CLASIF, header=FILA_CLASIF)

df_bz.columns = df_bz.columns.str.strip()
df_clasif.columns = df_clasif.columns.str.strip()

df_clasif.rename(columns={df_clasif.columns[0]: "NombreCuenta"}, inplace=True)
df_bz["NombreCuenta"] = df_bz["Nombre de Cuenta"].astype(str).str.strip()
df_clasif["NombreCuenta"] = df_clasif["NombreCuenta"].astype(str).str.strip()

df = pd.merge(df_bz, df_clasif, on="NombreCuenta", how="left")
df.dropna(subset=[LABEL_OBJETIVO], inplace=True)

df["NumeroCuenta"] = df["Número de cuenta"].astype(str).str.strip()
df["InputText"] = df["NumeroCuenta"] + " " + df["NombreCuenta"]

# === 2. Tokenización ===
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["InputText"])
X_seq = tokenizer.texts_to_sequences(df["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# === 3. Etiquetas ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[LABEL_OBJETIVO].astype(str))
num_classes = len(label_encoder.classes_)

# === 4. Cargar GloVe ===
print("📥 Cargando embeddings GloVe...")
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

# === 5. Crear modelo con embeddings preentrenados ===
print("🤖 Entrenando modelo con embeddings preentrenados...")
input_layer = Input(shape=(MAX_LEN,))
embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_LEN,
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

# === 6. Evaluación ===
y_pred = model.predict(X).argmax(axis=1)
print("\n📊 Reporte de clasificación:")
print(classification_report(y, y_pred, target_names=label_encoder.classes_))

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
model.save("modelo_con_glove.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ ¡Entrenamiento y guardado con GloVe completado!")
