import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# === 1. Cargar archivo ===
file_path = "Archivo Anual 2024.xlsx"
bz_df = pd.read_excel(file_path, sheet_name="BZ")
clasif_df = pd.read_excel(file_path, sheet_name="Clasificaciones")

# === 2. Preprocesamiento ===

# Asegúrate de identificar correctamente qué columnas usar
# Ejemplo: BZ tiene columna 'Nombre Cuenta', y Clasificaciones tiene 'Nombre Cuenta' + 'Cuenta SAT'
# Esto depende de tus nombres reales, cámbialos si es necesario

# Unimos la info de BZ con Clasificaciones
merged_df = pd.merge(bz_df, clasif_df, on='Nombre Cuenta', how='inner')

# Columnas:
# X = lo que queremos usar como entrada (ej. nombre, tipo de cuenta, descripción, combinados)
# y = la etiqueta: clave o nombre de cuenta SAT

# Creamos una columna combinada de texto para entrenar
merged_df["texto"] = merged_df["Nombre Cuenta"].astype(str) + " " + merged_df["Tipo"].astype(str)

# Etiqueta: la cuenta SAT
y = merged_df["Cuenta SAT"].astype(str)
X = merged_df["texto"]

# === 3. División de datos ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Pipeline de clasificación ===
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Entrenar modelo
pipeline.fit(X_train, y_train)

# === 5. Evaluar modelo ===
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# === 6. Guardar modelo entrenado ===
joblib.dump(pipeline, "clasificador_cuentas_sat.pkl")

print("✅ Modelo entrenado y guardado como 'clasificador_cuentas_sat.pkl'")
