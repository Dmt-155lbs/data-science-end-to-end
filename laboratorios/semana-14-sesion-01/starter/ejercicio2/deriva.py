"""El monitor de deriva de datos del servicio.

Guarda los ultimos `ventana` viajes que recibio el servicio y compara la
distribucion de distancia_km y de zona_origen con la de entrenamiento
(referencia_deriva.json), con el PSI del notebook del Ejercicio 1.

Despues de resolver, corre python comprobar.py hasta que los cinco pasos
digan OK, y despues python simular_semanas.py.
"""

import numpy as np

MINIMO = 0.0001


def proporciones_numericas(valores, cortes):
    """Proporcion de los valores en cada casilla. `cortes` son los limites
    interiores: con 9 cortes hay 10 casillas, la primera hasta el primer
    corte y la ultima desde el ultimo, abiertas. Los None no cuentan."""
    validos = [v for v in valores if v is not None]
    if not validos:
        return np.zeros(len(cortes) + 1)
    conteo, _ = np.histogram(validos, bins=[-np.inf] + list(cortes) + [np.inf])
    return conteo / conteo.sum()


def proporciones_categoricas(valores, categorias):
    """Proporcion de los valores en cada categoria, en el orden de
    `categorias`. Un valor que no esta en `categorias` no cuenta en ninguna
    casilla, pero si en el total."""
    valores = list(valores)
    if not valores:
        return np.zeros(len(categorias))
    return np.array([valores.count(categoria) for categoria in categorias]) / len(valores)


def psi(referencia, actual):
    """El PSI entre dos listas de proporciones, con MINIMO en las casillas
    vacias, como en el notebook."""
    referencia = np.clip(np.asarray(referencia, dtype=float), MINIMO, None)
    actual = np.clip(np.asarray(actual, dtype=float), MINIMO, None)
    return float(np.sum((actual - referencia) * np.log(actual / referencia)))


class MonitorDeDeriva:
    def __init__(self, referencia, ventana=300, umbral=0.15):
        self.referencia = referencia
        self.ventana = ventana
        self.umbral = umbral
        self.distancias = []
        self.zonas = []

    def registrar(self, viaje):
        """Agrega un viaje (un diccionario con distancia_km y zona_origen) y
        deja solo los ultimos `ventana`."""
        self.distancias.append(viaje.get("distancia_km"))
        self.zonas.append(viaje.get("zona_origen"))
        self.distancias = self.distancias[-self.ventana:]
        self.zonas = self.zonas[-self.ventana:]

    def estado(self):
        """Un diccionario con "viajes_en_ventana", "psi" (un diccionario con
        el PSI de distancia_km y de zona_origen, redondeado a 4 decimales) y
        "alerta": True solo si la ventana esta llena y alguno de los dos PSI
        es mayor que el umbral. Con la ventana vacia, "psi" es un diccionario
        vacio."""
        if not self.zonas:
            return {"viajes_en_ventana": 0, "psi": {}, "alerta": False}
        ref_distancia = self.referencia["distancia_km"]
        ref_zona = self.referencia["zona_origen"]
        valores = {
            "distancia_km": round(psi(ref_distancia["proporciones"],
                                      proporciones_numericas(self.distancias, ref_distancia["cortes"])), 4),
            "zona_origen": round(psi(ref_zona["proporciones"],
                                     proporciones_categoricas(self.zonas, ref_zona["categorias"])), 4),
        }
        llena = len(self.zonas) >= self.ventana
        alerta = llena and any(valor > self.umbral for valor in valores.values())
        return {"viajes_en_ventana": len(self.zonas), "psi": valores, "alerta": bool(alerta)}
