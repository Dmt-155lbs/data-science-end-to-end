import pandas as pd

viajes = pd.read_csv("../starter/viajes_taxi.csv")
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

# TODO 2: codifica zona_origen con one-hot (pd.get_dummies), con el
# prefijo "origen" y dtype=int. Imprime los nombres de las columnas
# nuevas.
origen_onehot = pd.get_dummies(viajes["zona_origen"], prefix="origen", dtype=int)
viajes = pd.concat([viajes, origen_onehot], axis=1)
print("columnas one-hot:", list(origen_onehot.columns))
print("cantidad one-hot:", origen_onehot.shape[1])

# TODO 3: codifica zona_origen de forma ordinal: un numero entero por
# zona, en orden alfabetico, en una columna origen_ordinal.
zonas_ordenadas = sorted(viajes["zona_origen"].unique())
mapa_zonas = {zona: i for i, zona in enumerate(zonas_ordenadas)}
viajes["origen_ordinal"] = viajes["zona_origen"].map(mapa_zonas)
print("codigos ordinales:", mapa_zonas)
print("columnas ordinal: 1 (origen_ordinal)")

# TODO 4: para cada columna original del dataset, decide si existe en el
# momento en que alguien pide el taxi (antes del viaje) e imprime la
# decision.
# 1 = existe al pedir el taxi, 0 = solo existe despues del viaje
disponible = {
    "id_viaje": 1,           # existe al pedir, pero es solo un identificador sin informacion
    "fecha_hora_inicio": 1,  # se conoce al pedir el taxi
    "zona_origen": 1,        # el pasajero indica donde esta
    "zona_destino": 1,       # el pasajero indica a donde va
    "distancia_km": 0,       # la medida real se conoce al terminar (al pedir solo hay estimacion)
    "duracion_min": 0,       # se conoce hasta que termina el viaje
    "pasajeros": 1,          # se declara al pedir el taxi
    "metodo_pago": 0,        # se registra al pagar, al final del viaje
    "tarifa": 0,             # es lo que queremos predecir
    "propina": 0,            # se da despues del viaje
}
for columna, existe in disponible.items():
    print(columna, "-> existe al pedir:", "si" if existe else "no")
print("no existen al pedir:", [c for c, e in disponible.items() if e == 0])
