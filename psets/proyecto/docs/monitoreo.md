# Monitoreo del modelo en producción

El servicio predice, al cierre de cada franja de 15 minutos, cuántas bicicletas salen de cada una de las 100 estaciones en los 30 minutos siguientes (ver [`despliegue.md`](despliegue.md)). Se vigila en dos capas: el **sistema** (¿el servicio responde, a tiempo y sin errores?) y el **modelo** (¿sus predicciones siguen siendo buenas?). La primera se ve en `/metricas` en el momento; la segunda necesita la verdad, que en este proyecto llega **30 minutos después** de cada predicción: las salidas reales de esa estación en esa media hora.

Cada predicción queda en `monitoreo/predicciones.jsonl` con su `id_prediccion`, el momento, la versión del modelo, la entrada completa (estación, fecha, franja y conteos) y la salida. [`monitoreo/desempeno.py`](../monitoreo/desempeno.py) la une con las salidas reales por estación, fecha y franja, y calcula las métricas del modelo por día y por versión.

## Las métricas que vigilo

| Métrica | Tipo | De dónde sale | Cuándo se conoce | Alerta si |
|---|---|---|---|---|
| Latencia p95 de `/predecir` | Sistema | `/metricas` (middleware) | En el momento | supera **100 ms**, el doble de lo normal (en la semana simulada: p50 de 30 ms y p95 de 51 ms). Con 100 estaciones por lote, el lote completo tardaría unos 10 s, todavía lejos del margen de 1 a 2 minutos, pero es señal de que algo cambió |
| Errores de servidor (5xx) | Sistema | `/metricas` | En el momento | hay **al menos uno** en un lote: una estación se queda sin predicción |
| Errores de validación (422) | Sistema | `/metricas` | En el momento | superan el **1%** de las peticiones de un día: el lote está mandando datos mal formados (un cambio en el pipeline de quien llama) |
| Predicciones por franja | Sistema | `predicciones.jsonl` | En el momento | hay **menos de 100** en una franja: hay estaciones sin predicción, aunque no haya errores |
| Disponibilidad (`/salud`) | Sistema | `/salud`, con `modelo_cargado` y `version_modelo` | En el momento | no responde, `modelo_cargado` es falso o la versión no es la de `produccion` en el registro |
| MAE diario | Modelo | `desempeno.py` (registro unido a las salidas reales) | 30 minutos después de cada predicción; se resume al cierre del día | el de la **semana** supera **1.102** (MAE de referencia 1.0278 + 2 desviaciones de la validación temporal, 0.037), o el de **un día** supera **1.25**. Un día suelto varía más que una semana: en la semana simulada fue de 0.84 (sábado 29) a 1.12 (jueves 27) |
| Costo asimétrico diario (subestimar cuesta 3, sobreestimar 1) | Modelo (negocio) | `desempeno.py` | Al cierre del día | el de la **semana** supera **2.30** (15% sobre el 2.00 de la validación temporal), o el de **un día** supera **2.60** (el peor día normal fue 2.35) |
| Predicción media frente a salidas reales medias | Modelo (proxy) | `desempeno.py`; la predicción media se puede calcular en el momento | La predicción media, en el momento; la comparación, 30 minutos después | la predicción media de un día se aleja más de **20%** de la del mismo día de la semana anterior: aviso temprano de drift en las entradas (otra demanda, una estación cerrada) |
| MAE por versión | Modelo | `desempeno.py` | Al cierre del día | la versión nueva queda peor que la anterior en los mismos días: volver a la anterior (ver "Versionado y registro" en `despliegue.md`) |

Los umbrales son iniciales: se ajustan con las primeras semanas en producción, para que una alerta signifique algo y no se ignore.

## Qué ve cada capa

Las métricas de sistema no ven ningún problema de calidad del modelo: si cambia la demanda (otra estación del año, una ciclovía nueva, un evento) el servicio sigue respondiendo en 50 ms sin errores y con la versión correcta, mientras predice cada vez peor. Eso solo lo muestran las métricas de modelo, y la verdad tarda 30 minutos. Mientras tanto, la predicción media por franja es el aviso más temprano.

## La semana de prueba, simulada

Con `python -m monitoreo.simular` mandé al servicio las 67,200 franjas del 25 al 31 de agosto (100 estaciones × 96 franjas por día), en orden, con la versión 1 en producción, y con `python -m monitoreo.desempeno` las uní con las salidas reales. Resultado en [`desempeno_por_dia.csv`](../monitoreo/desempeno_por_dia.csv):

- **Sistema:** 67,200 peticiones, 0 errores de validación, 0 de servidor, p50 de 30 ms y p95 de 51 ms; 100 predicciones en cada franja.
- **Modelo:** 67,100 predicciones unidas con su verdad (las 100 que faltan son la última franja del 31, que no tiene próximos 30 minutos). MAE de la semana **1.0254** y costo asimétrico **2.09**: sin alertas. El MAE es apenas menor que el 1.0278 de prueba porque el servicio recorta a 0 las predicciones negativas y redondea a 2 decimales. Por día, de 0.84 (sábado 29) a 1.12 (jueves 27).

## Deriva

Las métricas de modelo de arriba necesitan la verdad; la deriva de datos se puede vigilar en el momento, mirando solo lo que entra. [`servicio/deriva.py`](../servicio/deriva.py) guarda las últimas **9,600 franjas** que recibió el servicio (un día: 100 estaciones × 96 franjas) y calcula el PSI de dos features contra los días de entrenamiento (1 al 24 de agosto): `salidas_ultima_hora`, la entrada más importante del modelo, y `llegadas`. Las casillas son los cuantiles de entrenamiento sin repetir (los conteos tienen muchos ceros: quedan 7 casillas para la primera y 4 para la segunda). La referencia la escribe [`monitoreo/simular_deriva.py`](../monitoreo/simular_deriva.py) en `servicio/referencia_deriva.json`, y la ruta **`/deriva`** devuelve el PSI de la ventana actual y si hay alerta.

### La simulación

Con `python -m monitoreo.simular_deriva`: el PSI de cada uno de los 31 días de agosto sin cambiar nada ([`deriva_ventanas_normales.csv`](../monitoreo/deriva_ventanas_normales.csv)), y los 7 días de prueba con dos tipos de deriva ([`deriva_simulacion.csv`](../monitoreo/deriva_simulacion.csv), promedio de los 7 días):

- **Deriva de datos:** la demanda de todo el día cambia (lluvia o vacaciones, a la baja; un evento o una ciclovía nueva, al alza), en las entradas y en el objetivo por igual. Cambia lo que llega, no la relación entre lo reciente y lo que viene.
- **Deriva conceptual:** con la misma historia reciente salen más bicicletas en los 30 minutos siguientes (una promoción, una estación vecina cerrada que manda usuarios a estas). Las entradas no cambian; cambia lo que hay que predecir.

| Escenario | PSI de `salidas_ultima_hora` | PSI de `llegadas` | MAE | Costo asimétrico |
|---|---|---|---|---|
| Días normales (máximo de 31) | 0.0991 | 0.0549 | 1.1234 | 2.3483 |
| Días de prueba, sin deriva | 0.0299 | 0.0160 | 1.0255 | 2.0879 |
| Datos: demanda al 75% | 0.0415 | 0.0206 | 0.8799 | 1.6754 |
| Datos: demanda al 50% | 0.2076 | 0.1019 | 0.7275 | 1.2609 |
| Datos: demanda al 125% | 0.0815 | 0.0531 | 1.3118 | 2.8044 |
| Datos: demanda al 150% | 0.1826 | 0.1434 | 1.5690 | 3.4666 |
| Conceptual: salidas ×1.1 | 0.0299 | 0.0160 | 1.1136 | 2.4226 |
| Conceptual: salidas ×1.25 | 0.0299 | 0.0160 | 1.2689 | 2.9709 |
| Conceptual: salidas ×1.5 | 0.0299 | 0.0160 | 1.5699 | 3.9693 |

Lo que más mueve el PSI en un día normal es el **fin de semana**: los cuatro días normales con PSI más alto son sábados y domingos (0.099 el sábado 22), con menos demanda que la mezcla de entrenamiento. Un fin de semana ya es, para el monitor, una pequeña deriva de datos a la baja, y por eso una caída de 25% de la demanda no se distingue de un sábado.

### El umbral: 0.12

Alerta si el PSI de alguna de las dos features pasa **0.12** con la ventana llena ([`deriva_umbrales.csv`](../monitoreo/deriva_umbrales.csv)):

| Umbral | Falsas alarmas (31 días normales) | Detección, demanda al 50% | al 75% | al 125% | al 150% |
|---|---|---|---|---|---|
| 0.05 | 19% | 100% | 29% | 57% | 100% |
| 0.10 | 0% | 86% | 29% | 57% | 86% |
| **0.12** | **0%** | **71%** | **0%** | **29%** | **71%** |
| 0.15 | 0% | 57% | 0% | 14% | 57% |

0.10 tampoco tuvo falsas alarmas, pero queda pegado al peor día normal (0.099) y con 31 días no sé cuánto más puede moverse un sábado; 0.12 deja un margen de 20% y todavía ve la mayoría de los días con la demanda a la mitad o 50% más alta. Con una ventana de un día, la alerta llega al cierre del día, igual que el MAE diario; su ventaja es que no necesita la verdad, y que dice **qué** cambió.

### Qué no detecta esta alerta

- **La deriva conceptual, nunca:** con las salidas ×1.5 el PSI es el de un día normal (0.03) mientras el MAE sube de 1.03 a 1.57 y el costo asimétrico casi se duplica. Eso solo lo ve la alerta del MAE y del costo de arriba.
- **Cambios moderados de demanda:** al 75% no alerta nunca, y al 125%, solo 2 de 7 días. La subida es la que importa, porque **empeora** el modelo (MAE 1.31, costo 2.80, ya sobre su umbral semanal de 2.30); la bajada lo mejora (MAE 0.88), porque hay menos salidas que errar.
- **Cambios en lo que no vigila:** una estación nueva, otra capacidad, o el reparto de la demanda entre estaciones y horas con el mismo total. El monitor mira la distribución de dos conteos sobre todas las estaciones juntas.

**La regla con las dos alertas:** si el PSI alerta y el MAE no, es deriva de datos que el modelo absorbe (como un día de lluvia): se anota y se mira si sigue. Si el MAE alerta y el PSI no, es deriva conceptual: hay que reentrenar con datos recientes. Si alertan las dos, cambió la demanda y el modelo no la sigue (como un evento): se revisa qué pasó y se reentrena con datos que la incluyan.
