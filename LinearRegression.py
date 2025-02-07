import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv("/Users/mamadouwade/Documents/trainhouse_pricing.csv")

# Afficher les 5 premières lignes
print("Aperçu des données :")
print(df.head(), "\n")

# Détection et affichage des valeurs nulles
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values[missing_values > 0], "\n")

# Sélection des features et de la target
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'GarageCars']
target = 'SalePrice'

# Vérification de l'existence des colonnes sélectionnées
if not all(col in df.columns for col in features + [target]):
    raise ValueError("Une ou plusieurs colonnes sélectionnées n'existent pas dans le dataset.")

# Suppression des valeurs nulles
df = df.dropna(subset=features + [target])

# Détection et suppression des valeurs aberrantes
for col in features + [target]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Nombre de valeurs aberrantes pour {col}: {outliers.shape[0]}")
    
    # Suppression des outliers
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Séparation des données en X et y
X = df[features]
y = df[target]

# Division en ensemble d'entraînement et de test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Affichage des résultats
print("\nRésultats du modèle :")
print("Coefficients :", model.coef_)
print("Intercept :", model.intercept_)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Graphique des prédictions vs valeurs réelles
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Prédictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dashed', label="Idéal")
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Graphique des prédictions vs valeurs réelles")
plt.legend()
plt.show()

