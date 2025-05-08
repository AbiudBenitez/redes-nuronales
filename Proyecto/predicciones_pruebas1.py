# predecir_bz.py

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === CONFIGURACIÓN ===
MODEL_PATH = "modelo_mejorado.h5"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
MAX_LEN = 20

# === CARGAR COMPONENTES ===
print("📦 Cargando modelo y tokenizer...")
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# === PREDICCIÓN ===
def predecir_clasificacion(numero_cuenta, nombre_cuenta):
    texto = f"{str(numero_cuenta).strip()} {nombre_cuenta.strip()}"
    print(f"📝 Texto para predicción: '{texto}'")
    
    secuencia = tokenizer.texts_to_sequences([texto])
    entrada = pad_sequences(secuencia, maxlen=MAX_LEN, padding="post")
    
    pred = model.predict(entrada)
    clase_predicha = pred.argmax(axis=-1)[0]
    etiqueta = label_encoder.inverse_transform([clase_predicha])[0]
    
    return f"✅ Clasificación predicha: {etiqueta}"

# === EJEMPLO ===
if __name__ == "__main__":
    numero = input("🔢 Ingresa el número de cuenta: ")
    nombre = input("📘 Ingresa el nombre de cuenta: ")
    resultado = predecir_clasificacion(numero, nombre)
    print(resultado)
