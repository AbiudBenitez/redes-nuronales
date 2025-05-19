# === modelo_cnn_focal_glove.py ===
import os
import pandas as pd
import pickle
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import tensorflow.keras.backend as K

# === CONFIGURACION ===
CARPETA_DATOS = "files/dataset"
GLOVE_PATH = "files/glove.6B/glove.6B.100d.txt"
NOMBRE_HOJA = "BZ"
FILA_ENCABEZADO = 17
MAX_LEN = 20
NUM_WORDS = 5000
EMBEDDING_DIM = 100
LABEL_OBJETIVO = "Clasificación"

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

# === 1. CARGA Y PREPROCESAMIENTO DE DATOS ===
archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")]
df_completo = []

for archivo in archivos:
    try:
        ruta = os.path.join(CARPETA_DATOS, archivo)
        df = pd.read_excel(ruta, sheet_name=NOMBRE_HOJA, header=FILA_ENCABEZADO)
        df["__archivo_origen__"] = archivo
        df_completo.append(df)
    except Exception as e:
        print(f"Error leyendo {archivo}: {e}")

df = pd.concat(df_completo, ignore_index=True)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Nombre de Cuenta", LABEL_OBJETIVO])
df = df[df[LABEL_OBJETIVO].astype(str).str.strip() != ""]

for col in ["Número de cuenta", "Nombre de Cuenta"]:
    df[col] = df[col].astype(str).str.strip()
df["Nombre de Cuenta"] = df["Nombre de Cuenta"].apply(limpiar_texto)
df[LABEL_OBJETIVO] = df[LABEL_OBJETIVO].astype(str).str.strip()

# === 2. GENERAR INPUT DE TEXTO ===
df["InputText"] = (
    "CUENTA " + df["Número de cuenta"] +
    " NOMBRE " + df["Nombre de Cuenta"] +
    " ORIGEN " + df["__archivo_origen__"]
)

# === 3. TOKENIZACION Y SECUENCIAS ===
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["InputText"])
X_seq = tokenizer.texts_to_sequences(df["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# === 4. ENCODING DE ETIQUETAS ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[LABEL_OBJETIVO])
num_classes = len(label_encoder.classes_)

# === 5. DIVISION DE DATOS ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. CARGA DE EMBEDDINGS GLOVE ===
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

# === 7. MODELO CNN CON GLOVE + FOCAL LOSS ===
input_layer = Input(shape=(MAX_LEN,))
embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_LEN,
    trainable=False
)(input_layer)

x = Conv1D(128, 5, activation='relu')(embedding_layer)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss=focal_loss(gamma=2.0, alpha=0.25), metrics=["accuracy"])

# === 8. ENTRENAMIENTO ===
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, y_val))

# === 9. EVALUACION ===
y_pred = model.predict(X_val).argmax(axis=1)
labels_presentes = np.unique(y_val)
nombres_clases = label_encoder.inverse_transform(labels_presentes)

print(classification_report(y_val, y_pred, labels=labels_presentes, target_names=nombres_clases, zero_division=0))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=nombres_clases, yticklabels=nombres_clases, cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# === 10. GUARDADO ===
model.save("modelo_mejorado_glove.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
