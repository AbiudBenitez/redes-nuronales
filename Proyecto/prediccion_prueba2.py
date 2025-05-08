import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === CONFIGURACIÓN ===
ARCHIVO_INPUT = "files/dataset/Archivo Anual 2024.xlsx"   # Cambia a tu archivo a clasificar
HOJA_BZ = "BZ"
FILA_ENCABEZADO = 17
MAX_LEN = 20

# === 1. Cargar modelo y objetos ===
print("📦 Cargando modelo entrenado y objetos...")
model = load_model("modelo_mejorado.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === 2. Leer archivo nuevo ===
print("📄 Leyendo archivo nuevo...")
df = pd.read_excel(ARCHIVO_INPUT, sheet_name=HOJA_BZ, header=FILA_ENCABEZADO)
df.columns = df.columns.str.strip()

# Validación mínima
for col in ["Número de cuenta", "Nombre de Cuenta"]:
    if col not in df.columns:
        raise ValueError(f"❌ Falta columna requerida: {col}")

df["Número de cuenta"] = df["Número de cuenta"].astype(str).str.strip()
df["Nombre de Cuenta"] = df["Nombre de Cuenta"].astype(str).str.strip()

# === 3. Preparar texto de entrada ===
df["InputText"] = (
    "CUENTA " + df["Número de cuenta"] +
    " NOMBRE " + df["Nombre de Cuenta"]
)

X_seq = tokenizer.texts_to_sequences(df["InputText"])
X = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

# === 4. Predecir ===
print("🤖 Clasificando registros...")
y_pred = model.predict(X).argmax(axis=1)
df["Clasificación Predicha"] = label_encoder.inverse_transform(y_pred)

# === 5. Guardar resultado ===
output_path = "predicciones_clasificadas.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Clasificación completada. Archivo guardado como: {output_path}")
