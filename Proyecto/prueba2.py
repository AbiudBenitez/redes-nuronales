import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical

# === 1. Cargar archivo Excel ===
file_path = (
    "C:/Users/abiud/Documents/Code/redes-nuronales/Proyecto/Archivo Anual 2024.xlsx"
)
bz_df = pd.read_excel(file_path, sheet_name="BZ")
clasif_df = pd.read_excel(file_path, sheet_name="Clasificaciones")

# === 2. Unir datos por 'Nombre Cuenta' ===
merged_df = pd.merge(bz_df, clasif_df, on="Nombre de Cuenta", how="inner")

# === 3. Preparar texto y etiquetas ===
merged_df["texto"] = (
    merged_df["Nombre Cuenta"].astype(str) + " " + merged_df["Tipo"].astype(str)
)
X_text = merged_df["texto"].values
y_labels = merged_df["Cuenta SAT"].values

# === 4. Tokenizar texto ===
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
X_seq = tokenizer.texts_to_sequences(X_text)
X_pad = pad_sequences(X_seq, padding="post", maxlen=20)

# === 5. Codificar etiquetas ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_labels)
y_cat = to_categorical(y_encoded)

# === 6. Separar entrenamiento y prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_cat, test_size=0.2, random_state=42
)

# === 7. Crear modelo Keras ===
model = Sequential(
    [
        Embedding(input_dim=5000, output_dim=32, input_length=20),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dense(y_cat.shape[1], activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# === 8. Entrenar ===
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# === 9. Guardar modelo y tokenizador ===
model.save("modelo_cuentas_sat_keras.h5")

import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Modelo y herramientas guardados correctamente.")
