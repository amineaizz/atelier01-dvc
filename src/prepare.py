
import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

# Télécharger le dataset
dataset = fetch_ucirepo(id=17)

# Récupérer les données
X = dataset.data.features
y = dataset.data.targets

# Fusionner les données
df = pd.concat([X, y], axis=1)

# Créer le dossier data
os.makedirs("data", exist_ok=True)

# Sauvegarder en CSV
df.to_csv("data/breast_cancer.csv", index=False)

print("Dataset téléchargé avec succès !")
print("Dimensions :", df.shape)
print(df["Diagnosis"].value_counts())
