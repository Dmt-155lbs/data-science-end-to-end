import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

viajes = pd.read_csv("../starter/viajes_features.csv")

NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]

X = viajes[NUMERICAS + YA_LISTAS + CATEGORICAS]
y = viajes["tarifa"]

# TODO 1: divide X e y en entrenamiento (80%) y prueba (20%), con
# random_state=42.
X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Entrenamiento:", X_ent.shape, "Prueba:", X_prueba.shape)

# TODO 2: arma un Pipeline para las columnas NUMERICAS (imputar con la
# mediana, despues escalar) y otro para las CATEGORICAS (imputar con el
# valor mas frecuente, despues OneHotEncoder con handle_unknown="ignore" y
# sparse_output=False). Juntalos en un ColumnTransformer llamado
# "preparacion", con las columnas YA_LISTAS pasando sin cambios
# ("passthrough").

pipe_num = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler()),
])
pipe_cat = Pipeline([
    ("imputar", SimpleImputer(strategy="most_frequent")),
    ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preparacion = ColumnTransformer([
    ("num", pipe_num, NUMERICAS),
    ("cat", pipe_cat, CATEGORICAS),
    ("listas", "passthrough", YA_LISTAS),
])

# TODO 3: aprende la preparacion solo con entrenamiento y transforma
# entrenamiento y prueba. Imprime cuantas columnas quedan.
preparacion.fit(X_ent)
X_ent_prep = preparacion.transform(X_ent)
X_prueba_prep = preparacion.transform(X_prueba)
print("Forma entrenamiento preparado:", X_ent_prep.shape)
print("Forma prueba preparada:", X_prueba_prep.shape)
print("Columnas que quedan:", X_ent_prep.shape[1])

# TODO 4: imprime las medianas que aprendio el imputador de numericas
# (atributo statistics_) junto a las medianas de entrenamiento y las de
# todo el dataset.
med_aprendidas = preparacion.named_transformers_["num"].named_steps["imputar"].statistics_
print("Medianas aprendidas por el imputador:", dict(zip(NUMERICAS, med_aprendidas.tolist())))
print("Medianas de entrenamiento (pandas):", X_ent[NUMERICAS].median().to_dict())
print("Medianas de todo el dataset (pandas):", X[NUMERICAS].median().to_dict())
print("Coinciden con entrenamiento:", bool((med_aprendidas == X_ent[NUMERICAS].median().values).all()))
print("Coinciden con todo el dataset:", bool((med_aprendidas == X[NUMERICAS].median().values).all()))

# TODO 5: transforma el viaje nuevo de abajo, que sale de una zona que el
# entrenamiento nunca vio, e imprime cuantas columnas de zona_origen
# quedan en 1.
nuevo = pd.DataFrame([{
    "distancia_km": 9.0, "pasajeros": 1, "hora_sin": 0.5, "hora_cos": -0.866,
    "es_fin_de_semana": 0, "es_hora_pico": 1, "es_tarifa_fija": 0,
    "zona_origen": "Bronx", "zona_destino": "Midtown", "metodo_pago": "tarjeta",
}])

nuevo_prep = preparacion.transform(nuevo)
nombres = preparacion.get_feature_names_out()
print("Forma del viaje nuevo preparado:", nuevo_prep.shape)
cols_origen = [i for i, n in enumerate(nombres) if n.startswith("cat__zona_origen_")]
print("Columnas de zona_origen en 1:", int(nuevo_prep[0, cols_origen].sum()))
print("Columnas de zona_origen:", [nombres[i] for i in cols_origen])
print("Valores zona_origen del viaje nuevo:", nuevo_prep[0, cols_origen])
print("Zonas de origen vistas:", list(X_ent["zona_origen"].dropna().unique()))
