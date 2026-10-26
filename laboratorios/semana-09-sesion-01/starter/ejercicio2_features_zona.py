"""Ejercicio 2: zona, codificacion y disponibilidad en viajes_taxi.csv.

Corre el script con:

    python3 ejercicio2_features_zona.py

Al final se imprime una comprobacion por cada TODO. No la modifiques:
cuando las cuatro digan OK, pasa a las preguntas del README.
"""

import pandas as pd

viajes = pd.read_csv("viajes_taxi.csv")
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()

ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

# TODO 1: crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
# va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
# JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
# y maxima de esos viajes.
desde_jfk = (viajes["zona_origen"] == "JFK Airport") & viajes["zona_destino"].isin(ZONAS_MANHATTAN)
hacia_jfk = viajes["zona_origen"].isin(ZONAS_MANHATTAN) & (viajes["zona_destino"] == "JFK Airport")
viajes["es_tarifa_fija"] = (desde_jfk | hacia_jfk).astype(int)
fijos = viajes[viajes["es_tarifa_fija"] == 1]
print("viajes con es_tarifa_fija = 1:", len(fijos))
print("tarifa minima:", fijos["tarifa"].min())
print("tarifa maxima:", fijos["tarifa"].max())

# TODO 2: codificado, el resultado de codificar zona_origen con one-hot
# (pd.get_dummies), con el prefijo "origen" y dtype=int. Imprime los
# nombres de las columnas nuevas.
codificado = pd.get_dummies(viajes["zona_origen"], prefix="origen", dtype=int)
print("columnas one-hot:", list(codificado.columns))
print("cantidad one-hot:", codificado.shape[1])

# TODO 3: codifica zona_origen de forma ordinal: un numero entero por
# zona, en orden alfabetico, en una columna origen_ordinal.
zonas_ordenadas = sorted(viajes["zona_origen"].unique())
mapa_zonas = {zona: i for i, zona in enumerate(zonas_ordenadas)}
viajes["origen_ordinal"] = viajes["zona_origen"].map(mapa_zonas)
print("codigos ordinales:", mapa_zonas)

# TODO 4: disponible_al_pedir, un diccionario con cada columna original
# del dataset y True o False segun exista en el momento en que alguien
# pide el taxi (antes del viaje). Imprime la decision de cada una.
disponible_al_pedir = {
    "id_viaje": True,            # existe al pedir, pero es solo un identificador sin informacion
    "fecha_hora_inicio": True,   # se conoce al pedir el taxi
    "zona_origen": True,         # el pasajero indica donde esta
    "zona_destino": True,        # el pasajero indica a donde va
    "distancia_km": True,        # la app la estima con la ruta entre origen y destino
    "duracion_min": False,       # se conoce hasta que termina el viaje
    "pasajeros": True,           # se declara al pedir el taxi
    "metodo_pago": False,        # se registra al pagar, al final del viaje
    "tarifa": False,             # es lo que queremos predecir
    "propina": False,            # se da despues del viaje
}
for columna, existe in disponible_al_pedir.items():
    print(columna, "-> existe al pedir:", "si" if existe else "no")


def comprobar(numero, condicion):
    if condicion:
        print(f"TODO {numero}: OK")
    else:
        print(f"TODO {numero}: revisa")


print()
print("Comprobacion")
comprobar(1, "es_tarifa_fija" in viajes.columns and viajes["es_tarifa_fija"].sum() == 300)
comprobar(2, codificado is not None
          and len([columna for columna in codificado.columns if columna.startswith("origen_")]) == 8)
comprobar(3, "origen_ordinal" in viajes.columns
          and viajes.loc[viajes["zona_origen"] == "Brooklyn", "origen_ordinal"].iloc[0] == 0
          and viajes["origen_ordinal"].max() == 7)
comprobar(4, disponible_al_pedir is not None
          and disponible_al_pedir.get("duracion_min") is False
          and disponible_al_pedir.get("propina") is False
          and disponible_al_pedir.get("distancia_km") is True)
