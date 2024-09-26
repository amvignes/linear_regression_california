# Importer les bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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

# Initialiser le modèle de Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)

# Entraîner le modèle
rf_model.fit(X_train_scaled, y_train)

# Prédire les prix sur l'ensemble de test
y_pred = rf_model.predict(X_test_scaled)

# Calculer l'erreur quadratique moyenne (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Erreur quadratique moyenne (Random Forest): {mse}')

# Tracer les valeurs prédites contre les valeurs réelles
plt.scatter(y_test, y_pred)
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Random Forest: valeurs réelles vs prédites")
plt.show()
