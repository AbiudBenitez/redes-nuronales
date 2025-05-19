import pickle

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("👉 Clases guardadas en el LabelEncoder:")
print(le.classes_)
