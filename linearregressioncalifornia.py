# Importer les bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Charger le dataset California Housing
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les données (mettre les variables sur la même échelle)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser le modèle de régression régularisée Ridge
ridge_model = Ridge(alpha=1.0)  # L'alpha contrôle la régularisation, plus il est élevé, plus la régularisation est forte.

# Entraîner le modèle
ridge_model.fit(X_train_scaled, y_train)

# Prédire les prix sur l'ensemble de test
y_pred = ridge_model.predict(X_test_scaled)

# Calculer l'erreur quadratique moyenne (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Erreur quadratique moyenne (Ridge): {mse}')

# Afficher les coefficients du modèle
print("Coefficients du modèle Ridge :", ridge_model.coef_)
print("Intercept :", ridge_model.intercept_)

# Tracer les valeurs prédites contre les valeurs réelles
plt.scatter(y_test, y_pred)
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Régression Ridge: valeurs réelles vs prédites")
plt.show()
