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

