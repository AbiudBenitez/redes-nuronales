# === modelo_cnn_multisalida.py ===
import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

# === CONFIGURACION ===
CARPETA_DATOS = "files/dataset"
GLOVE_PATH = "files/glove.6B/glove.6B.100d.txt"
NOMBRE_HOJA = "BZ"
FILA_ENCABEZADO = 17
MAX_LEN = 20
NUM_WORDS = 5000
EMBEDDING_DIM = 100
LABEL_CLASIFICACION = "Clasificación"
LABEL_PRPNR = "PR / PNR"

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

# === CARGA DE ARCHIVOS ===
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

# === UNIFICACION Y LIMPIEZA ===
df = pd.concat(df_completo, ignore_index=True)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Nombre de Cuenta", LABEL_CLASIFICACION, LABEL_PRPNR])
df = df[df[LABEL_CLASIFICACION].astype(str).str.strip() != ""]

# === NORMALIZACION ===
df["Numero"] = df["Número de cuenta"].astype(str).str.strip()
df["Nombre"] = df["Nombre de Cuenta"].astype(str).apply(limpiar_texto)
df[LABEL_CLASIFICACION] = df[LABEL_CLASIFICACION].astype(str).str.strip()
df[LABEL_PRPNR] = df[LABEL_PRPNR].str.strip().map({"PR": 1, "PNR": 0})

# === INPUT DE TEXTO ===
df["InputText"] = "CUENTA " + df["Numero"] + " NOMBRE " + df["Nombre"] + " ORIGEN " + df["__archivo_origen__"]

# === TOKENIZACION ===
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["InputText"])
X_seq = tokenizer.texts_to_sequences(df["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# === ENCODING DE ETIQUETAS ===
label_encoder = LabelEncoder()
y_clasif = label_encoder.fit_transform(df[LABEL_CLASIFICACION])
y_prpnr = df[LABEL_PRPNR].values.astype(np.float32)
num_classes = len(label_encoder.classes_)

# === DIVISION ===
X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    X, y_clasif, y_prpnr, test_size=0.2, random_state=42)

# === CARGA DE EMBEDDINGS GLOVE ===
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

# === MODELO MULTISALIDA ===
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

input_layer = Input(shape=(MAX_LEN,))
embedding_layer = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                             weights=[embedding_matrix], input_length=MAX_LEN, trainable=False)(input_layer)

x = Conv1D(128, 5, activation='relu')(embedding_layer)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

salida_clasificacion = Dense(num_classes, activation="softmax", name="clasificacion_output")(x)
salida_prpnr = Dense(1, activation="sigmoid", name="prpnr_output")(x)

model = Model(inputs=input_layer, outputs=[salida_clasificacion, salida_prpnr])
model.compile(optimizer="adam",
              loss={"clasificacion_output": focal_loss(), "prpnr_output": "binary_crossentropy"},
              loss_weights={"clasificacion_output": 1.0, "prpnr_output": 0.5},
              metrics={"clasificacion_output": "accuracy", "prpnr_output": "accuracy"})

# === ENTRENAMIENTO ===
model.fit(X_train, {"clasificacion_output": y1_train, "prpnr_output": y2_train},
          epochs=30, batch_size=16,
          validation_data=(X_val, {"clasificacion_output": y1_val, "prpnr_output": y2_val}))

# === GUARDADO ===
model.save("modelo_multisalida_glove.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Modelo multisalida entrenado y guardado correctamente.")
