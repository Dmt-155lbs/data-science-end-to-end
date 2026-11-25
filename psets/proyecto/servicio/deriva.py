"""El monitor de deriva de datos del servicio (adaptado del Ejercicio 2 de la
Semana 14, Sesion 1).

Guarda las ultimas `ventana` franjas que recibio el servicio (por defecto un
dia: 100 estaciones x 96 franjas) y compara la distribucion de cada feature
vigilada con la de entrenamiento (servicio/referencia_deriva.json, que escribe
monitoreo/simular_deriva.py) con el PSI. Todas las features vigiladas son
numericas: sus casillas son los cuantiles de entrenamiento.
"""

import numpy as np

MINIMO = 0.0001
FEATURES_VIGILADAS = ["salidas_ultima_hora", "llegadas"]
VENTANA = 9600
UMBRAL = 0.12  # sobre el peor dia normal de agosto (0.099); ver docs/monitoreo.md


def cortes_de_referencia(valores, casillas=10):
    """Los limites interiores de las casillas: los cuantiles de entrenamiento,
    sin repetidos (los conteos tienen muchos ceros, y varios deciles caen en
    el mismo valor)."""
    valores = np.asarray([v for v in valores if v is not None and not np.isnan(v)], dtype=float)
    cortes = np.unique(np.quantile(valores, np.linspace(0, 1, casillas + 1))[1:-1])
    return [float(c) for c in cortes]


def proporciones_numericas(valores, cortes):
    """Proporcion de los valores en cada casilla; las casillas son
    (-inf, c1], (c1, c2], ..., (ck, inf). Los conteos son enteros, asi que se
    cierran por la derecha para que el 0 quede solo en su casilla. Los None
    no cuentan."""
    validos = np.asarray([v for v in valores if v is not None and not np.isnan(v)], dtype=float)
    if len(validos) == 0:
        return np.zeros(len(cortes) + 1)
    conteo = np.bincount(np.searchsorted(cortes, validos, side="left"), minlength=len(cortes) + 1)
    return conteo / conteo.sum()


def psi(referencia, actual):
    referencia = np.clip(np.asarray(referencia, dtype=float), MINIMO, None)
    actual = np.clip(np.asarray(actual, dtype=float), MINIMO, None)
    return float(np.sum((actual - referencia) * np.log(actual / referencia)))


def construir_referencia(datos, features=FEATURES_VIGILADAS):
    """Cortes y proporciones de cada feature en los datos de entrenamiento."""
    referencia = {}
    for feature in features:
        cortes = cortes_de_referencia(datos[feature])
        referencia[feature] = {
            "cortes": cortes,
            "proporciones": [float(p) for p in proporciones_numericas(datos[feature], cortes)],
        }
    return referencia


class MonitorDeDeriva:
    def __init__(self, referencia, ventana=VENTANA, umbral=UMBRAL):
        self.referencia = referencia
        self.ventana = ventana
        self.umbral = umbral
        self.valores = {feature: [] for feature in referencia}

    def registrar(self, franja):
        """Agrega una franja (un diccionario con las features vigiladas) y deja
        solo las ultimas `ventana`."""
        for feature, lista in self.valores.items():
            lista.append(franja.get(feature))
            del lista[:-self.ventana]

    def viajes_en_ventana(self):
        return len(next(iter(self.valores.values()), []))

    def estado(self):
        """Franjas en la ventana, el PSI de cada feature y la alerta: solo con
        la ventana llena y algun PSI sobre el umbral."""
        n = self.viajes_en_ventana()
        if n == 0:
            return {"franjas_en_ventana": 0, "umbral": self.umbral, "psi": {}, "alerta": False}
        valores = {
            feature: round(psi(ref["proporciones"], proporciones_numericas(self.valores[feature], ref["cortes"])), 4)
            for feature, ref in self.referencia.items()
        }
        alerta = n >= self.ventana and any(v > self.umbral for v in valores.values())
        return {"franjas_en_ventana": n, "umbral": self.umbral, "psi": valores, "alerta": bool(alerta)}
