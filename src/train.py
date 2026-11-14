
import pandas as pd
import yaml
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Lire les paramètres
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Charger le dataset
df = pd.read_csv("data/breast_cancer.csv")

# Séparer les variables et la cible
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"],
    stratify=y
)

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    random_state=params["train"]["random_state"]
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)

# Calculer les métriques
metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "f1_score": float(f1_score(y_test, predictions, pos_label="M"))
}

# Créer les dossiers
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, "models/model.pkl")

# Sauvegarder les métriques
with open("reports/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Entraînement terminé !")
print(metrics)
