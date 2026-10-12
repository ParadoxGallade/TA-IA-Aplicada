"""Implementación de modelos."""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def entrenar_regresion_lineal(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_arbol_regresion(X_train, y_train, max_depth=5):
    modelo = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo
