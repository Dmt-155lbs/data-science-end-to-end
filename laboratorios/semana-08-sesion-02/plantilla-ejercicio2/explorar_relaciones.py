import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# TODO 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
limpios = viajes[(viajes["tarifa"] != 999.99) & ~(viajes["duracion_min"] < 0)]
print("Filas que quedan:", len(limpios))

# TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general y
# cuentalos con un filtro.
plt.scatter(limpios["distancia_km"], limpios["tarifa"], s=8, alpha=0.5)
plt.xlabel("distancia_km")
plt.ylabel("tarifa")
plt.savefig("figuras/distancia_vs_tarifa.png")
plt.close()

# Filtro: tarifa plana de 70.00, que no depende de la distancia (todos
# tocan JFK Airport) y forma una linea horizontal fuera de la tendencia.
tarifa_plana = limpios[limpios["tarifa"] == 70.0]
print("Viajes con tarifa plana 70:", len(tarifa_plana))

# TODO 3: propina promedio y cantidad de viajes con propina 0, por
# metodo de pago (groupby + agg).
por_pago = limpios.groupby("metodo_pago")["propina"].agg(
    propina_promedio="mean", viajes_propina_0=lambda x: (x == 0).sum()
)
print(por_pago)

# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
# con la tarifa.
columnas = ["distancia_km", "duracion_min", "pasajeros", "propina"]
print(limpios[columnas].corrwith(limpios["tarifa"]))

# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
print("Pasajeros vacio:", limpios["pasajeros"].isna().sum())
print("Pasajeros en 0:", (limpios["pasajeros"] == 0).sum())
