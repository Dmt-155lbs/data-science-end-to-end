"""Prepara el CSV de flujo por estacion x dia x franja de 15 min (Ecobici, agosto 2026).

Replica de forma simplificada la logica del proyecto (dbt + Spark):
limpieza de viajes -> grilla densa -> rezagos y objetivo.
Uso: python preparar_datos.py
"""
import os
import random

import duckdb

PERIODO = "2026-08"
N_ESTACIONES = 100
SEMILLA = 42

# rutas relativas a este script
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", "..", ".."))  # data-science-end-to-end
DATOS = os.path.abspath(os.path.join(RAIZ, "..", "..", "datos_ecobici"))  # carpeta fuera del repo
VIAJES = os.path.join(DATOS, "raw", "viajes", "public_data_web_2026-08_2.csv")
GBFS = os.path.join(DATOS, "raw", "gbfs", "station_information.json")
SALIDA = os.path.join(RAIZ, "data", "processed", "flujo_estaciones_2026_08.csv")
os.makedirs(os.path.dirname(SALIDA), exist_ok=True)


def sql_path(p):
    return p.replace("\\", "/")


con = duckdb.connect()

# 1. lectura cruda: todo texto, nombres en minuscula y con alias (como ingesta_viajes.py)
con.execute(f"""
    CREATE TABLE raw AS
    SELECT * FROM read_csv('{sql_path(VIAJES)}', header=true, all_varchar=true,
                           normalize_names=true)
""")
cols = [r[0] for r in con.execute("DESCRIBE raw").fetchall()]
alias = {"ciclo_estacionarribo": "ciclo_estacion_arribo", "ciclo_estacionretiro": "ciclo_estacion_retiro"}
sel = ", ".join(f'"{c}" AS {alias.get(c, c)}' for c in cols)
con.execute(f"CREATE TABLE raw2 AS SELECT {sel} FROM raw")

# 2. tipado (stg_viajes): fecha DD/MM/YYYY o YYYY-MM-DD; hora local America/Mexico_City (sin zona)
con.execute("""
    CREATE TABLE stg AS
    SELECT
        nullif(trim(bici), '') AS bici,
        nullif(trim(ciclo_estacion_retiro), '') AS est_retiro,
        nullif(trim(ciclo_estacion_arribo), '') AS est_arribo,
        coalesce(try_strptime(trim(fecha_retiro), '%d/%m/%Y'), try_strptime(trim(fecha_retiro), '%Y-%m-%d')) AS f_ret,
        coalesce(try_strptime(trim(fecha_arribo), '%d/%m/%Y'), try_strptime(trim(fecha_arribo), '%Y-%m-%d')) AS f_arr,
        try_cast(trim(hora_retiro) AS TIME) AS h_ret,
        try_cast(trim(hora_arribo) AS TIME) AS h_arr,
        md5(concat_ws('|', coalesce(genero_usuario,''), coalesce(edad_usuario,''), coalesce(bici,''),
            coalesce(ciclo_estacion_retiro,''), coalesce(fecha_retiro,''), coalesce(hora_retiro,''),
            coalesce(ciclo_estacion_arribo,''), coalesce(fecha_arribo,''), coalesce(hora_arribo,''))) AS huella
    FROM raw2
""")
con.execute("""
    CREATE TABLE tipado AS
    SELECT *, CAST(f_ret AS DATE) + h_ret AS retiro_ts, CAST(f_arr AS DATE) + h_arr AS arribo_ts
    FROM stg
""")

# 3. limpieza (int_viajes): duplicados, fecha invalida, llegada antes de salida,
#    duracion < 1 min, salida anterior al periodo (1 dia de tolerancia)
con.execute(f"""
    CREATE TABLE viajes AS
    WITH num AS (
        SELECT *,
            date_diff('second', retiro_ts, arribo_ts) / 60.0 AS dur_min,
            row_number() OVER (PARTITION BY huella ORDER BY retiro_ts) AS n_exacta,
            row_number() OVER (PARTITION BY bici, retiro_ts ORDER BY arribo_ts DESC NULLS LAST) AS n_logica
        FROM tipado
    )
    SELECT est_retiro, est_arribo, retiro_ts, arribo_ts
    FROM num
    WHERE n_exacta = 1 AND n_logica = 1
      AND retiro_ts IS NOT NULL
      AND NOT coalesce(arribo_ts < retiro_ts, false)
      AND NOT coalesce(dur_min >= 0 AND dur_min < 1, false)
      AND CAST(retiro_ts AS DATE) >= DATE '{PERIODO}-01' - INTERVAL 1 DAY
""")

# 4. catalogo GBFS (cruce por short_name = codigo de estacion en los viajes)
con.execute(f"""
    CREATE TABLE catalogo AS
    SELECT s.short_name AS estacion_codigo, s.name AS nombre, s.lat AS latitud,
           s.lon AS longitud, s.capacity AS capacidad
    FROM (SELECT unnest(data.stations) AS s FROM read_json('{sql_path(GBFS)}'))
""")

# 5. muestra reproducible de estaciones (con salidas en el mes y presentes en el catalogo)
candidatas = [r[0] for r in con.execute(f"""
    SELECT DISTINCT v.est_retiro FROM viajes v JOIN catalogo c ON c.estacion_codigo = v.est_retiro
    WHERE strftime(v.retiro_ts, '%Y-%m') = '{PERIODO}' ORDER BY 1
""").fetchall()]
muestra = random.Random(SEMILLA).sample(candidatas, N_ESTACIONES)
con.execute("CREATE TABLE muestra (estacion_codigo VARCHAR)")
con.executemany("INSERT INTO muestra VALUES (?)", [(e,) for e in muestra])
print(f"estaciones candidatas: {len(candidatas)} | en la muestra: {len(muestra)}")

# 6. grilla densa estacion x dia x franja + conteos (fct_flujo_estacion)
con.execute(f"""
    CREATE TABLE flujo AS
    WITH dias AS (
        SELECT CAST(d AS DATE) AS fecha
        FROM generate_series(DATE '{PERIODO}-01',
                             last_day(DATE '{PERIODO}-01'), INTERVAL 1 DAY) t(d)
    ),
    grilla AS (
        SELECT m.estacion_codigo, d.fecha, f.franja
        FROM muestra m CROSS JOIN dias d CROSS JOIN range(96) f(franja)
    ),
    sal AS (
        SELECT est_retiro AS estacion_codigo, CAST(retiro_ts AS DATE) AS fecha,
               hour(retiro_ts) * 4 + minute(retiro_ts) // 15 AS franja, count(*) AS salidas
        FROM viajes GROUP BY ALL
    ),
    lle AS (
        SELECT est_arribo AS estacion_codigo, CAST(arribo_ts AS DATE) AS fecha,
               hour(arribo_ts) * 4 + minute(arribo_ts) // 15 AS franja, count(*) AS llegadas
        FROM viajes WHERE est_arribo IS NOT NULL AND arribo_ts IS NOT NULL GROUP BY ALL
    )
    SELECT g.estacion_codigo, g.fecha, g.franja,
           coalesce(s.salidas, 0) AS salidas, coalesce(l.llegadas, 0) AS llegadas
    FROM grilla g
    LEFT JOIN sal s USING (estacion_codigo, fecha, franja)
    LEFT JOIN lle l USING (estacion_codigo, fecha, franja)
""")

# 7. atributos, rezagos y objetivo (construir_obt.py). Como la grilla es densa,
#    desplazar N filas = desplazar N x 15 min. Los bordes del mes quedan NULL en
#    lags; en ventanas (suma) las parciales suman lo disponible y las vacias son NULL.
con.execute(f"""
    COPY (
        SELECT
            f.estacion_codigo, c.nombre, c.latitud, c.longitud, c.capacidad,
            f.fecha, f.franja // 4 AS hora, f.franja,
            isodow(f.fecha) AS dia_semana,
            isodow(f.fecha) >= 6 AS es_fin_de_semana,
            (f.franja // 4 BETWEEN 7 AND 9) OR (f.franja // 4 BETWEEN 17 AND 19) AS es_hora_pico,
            f.salidas, f.llegadas, f.llegadas - f.salidas AS flujo_neto,
            lag(f.salidas, 1) OVER w AS salidas_lag_15m,
            sum(f.salidas) OVER (w ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS salidas_ultima_hora,
            lag(f.salidas, 96) OVER w AS salidas_mismo_momento_ayer,
            sum(f.salidas) OVER (w ROWS BETWEEN 1 FOLLOWING AND 2 FOLLOWING) AS y_salidas_prox_30m
        FROM flujo f LEFT JOIN catalogo c ON c.estacion_codigo = f.estacion_codigo
        WINDOW w AS (PARTITION BY f.estacion_codigo ORDER BY f.fecha, f.franja)
        ORDER BY f.estacion_codigo, f.fecha, f.franja
    ) TO '{sql_path(SALIDA)}' (HEADER, DELIMITER ',')
""")

# 8. chequeos
tabla = f"read_csv('{sql_path(SALIDA)}', header=true, types={{'estacion_codigo': 'VARCHAR'}})"
print("filas:", con.execute(f"SELECT count(*) FROM {tabla}").fetchone()[0], "| esperado:", N_ESTACIONES * 31 * 96)
print("suma salidas CSV:", con.execute(f"SELECT sum(salidas) FROM {tabla}").fetchone()[0])
print("viajes limpios con salida en la muestra y en el mes:", con.execute(f"""
    SELECT count(*) FROM viajes WHERE est_retiro IN (SELECT estacion_codigo FROM muestra)
    AND strftime(retiro_ts, '%Y-%m') = '{PERIODO}'""").fetchone()[0])
print("viajes crudos:", con.execute("SELECT count(*) FROM raw2").fetchone()[0],
      "| limpios:", con.execute("SELECT count(*) FROM viajes").fetchone()[0])
df = con.execute(f"SELECT * FROM {tabla}").df()
nulos = df.isna().sum()
print("NULOS por columna:\n", nulos[nulos > 0].to_string() if (nulos > 0).any() else "ninguno")
print(df["y_salidas_prox_30m"].describe())
print("tamano CSV (MB):", round(os.path.getsize(SALIDA) / 1e6, 2))
