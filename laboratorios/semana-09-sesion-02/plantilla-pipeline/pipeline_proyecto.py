import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# El archivo de la sesion pasada, con las 2 features ya agregadas
# (periodo_demanda y salidas_hora_por_capacidad).
# estacion_codigo se lee como texto por los codigos dobles como "271-272".
datos = pd.read_csv(
    "../../semana-09-sesion-01/data/flujo_estaciones_features_2026_08.csv",
    dtype={"estacion_codigo": str},
)

OBJETIVO = "y_salidas_prox_30m"

# Sin objetivo no se puede entrenar: la ultima franja de cada estacion no
# tiene "proximos 30 min". Se quitan esas filas antes de dividir.
datos = datos.dropna(subset=[OBJETIVO])

# Solo columnas que existen al momento de predecir (nada de y_*, ni
# identificadores como nombre o fecha). Los NaN de los rezagos al inicio de
# cada serie los imputa el pipeline.
NUMERICAS = [
    "salidas", "llegadas", "salidas_lag_15m", "salidas_ultima_hora",
    "salidas_mismo_momento_ayer", "salidas_hora_por_capacidad",
    "capacidad", "hora", "dia_semana",
]
# estacion_codigo no entra: son 100 estaciones y un one-hot agregaria 100
# columnas; la capacidad y los rezagos ya describen a cada estacion.
# (Si entrara, se dejaria handle_unknown="ignore" por estaciones nuevas.)
CATEGORICAS = ["periodo_demanda"]
# es_fin_de_semana es booleana: se pasa como 0/1, ya esta lista.
YA_LISTAS = ["es_fin_de_semana"]

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
y = datos[OBJETIVO]

# Las filas tienen fecha y el modelo predice el futuro, asi que se divide
# por fecha: entrenamiento dias 1-24 de agosto, prueba dias 25-31. Un corte
# al azar mezclaria franjas vecinas (los rezagos y el objetivo se traslapan
# entre franjas contiguas) y dejaria pasar informacion del futuro al
# entrenamiento.
dia = pd.to_datetime(datos["fecha"]).dt.day
es_ent = dia <= 24
X_ent, X_prueba = X[es_ent], X[~es_ent]
y_ent, y_prueba = y[es_ent], y[~es_ent]
print("Filas entrenamiento:", len(X_ent), "Filas prueba:", len(X_prueba))

preparacion = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ]), NUMERICAS),
    ("categoricas", Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAS),
    ("ya_listas", "passthrough", YA_LISTAS),
])

X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
print(list(preparacion.get_feature_names_out()))

# Verificacion: las medianas aprendidas son las del entrenamiento.
medianas = preparacion.named_transformers_["numericas"].named_steps["imputar"].statistics_
print("Medianas iguales a las de entrenamiento:",
      bool((medianas == X_ent[NUMERICAS].median().values).all()))
print("NaN restantes:", int(pd.isna(X_ent_listo).sum() + pd.isna(X_prueba_listo).sum()))
