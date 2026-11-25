# Semana 14, Sesión 1: Monitoreo en producción II: deriva de datos y deriva conceptual

> Objetivo de la sesión: distinguir deriva de datos de deriva conceptual en el modelo de tu proyecto, y diseñar una alerta que detecte cuál de las dos lo está degradando.

**Lectura previa de esta sesión:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 9: Continual Learning and Test in Production.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El Ejercicio 1 es un notebook (simula la deriva y elige el umbral); el Ejercicio 2, un script del servicio (la alerta queda ahí, corriendo con cada predicción).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la deriva de datos y qué es la deriva conceptual? Da un ejemplo de cada una.**

La **deriva de datos** (data drift) es un cambio en la distribución de las entradas que recibe el modelo, sin que cambie la relación entre esas entradas y lo que se predice: el modelo ve otra mezcla de casos, pero cada caso se resuelve igual. Ejemplo: un modelo de tarifas entrenado con viajes de días laborables que en vacaciones recibe muchos más viajes al aeropuerto. La **deriva conceptual** (concept drift) es un cambio en la relación entre las entradas y el objetivo, aunque las entradas sigan iguales: lo que el modelo aprendió deja de valer. El artículo pone el ejemplo de una campaña de marketing que cambia cómo se comportan los clientes; en taxis sería que la ciudad suba la tarifa por taxímetro: los mismos viajes, otro precio.

**¿Por qué el artículo dice que hay que monitorear las dos, y no solo el desempeño del modelo?**

Porque el desempeño solo se puede medir cuando llega la verdad, y eso puede tardar (o no llegar nunca para todos los casos); mientras tanto el modelo ya está decidiendo mal. Vigilar la distribución de las entradas da un aviso temprano, sin esperar la verdad, y además ayuda a saber **por qué** cayó el desempeño: si cambiaron los datos o si cambió la relación. Y al revés, la deriva conceptual no se ve en las entradas, solo en el desempeño. Con las dos se distingue el tipo de problema, y eso decide qué hacer (esperar, revisar el pipeline, reentrenar).

## Pregunta de apertura

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

Cambió la relación entre los viajes y la tarifa, no los viajes: es deriva conceptual. Los viajes de noviembre tienen las mismas zonas, distancias y horas que los de octubre, pero desde el 16 de noviembre el mismo viaje se cobra distinto (por ejemplo, un aumento de la tarifa por taxímetro), así que lo que el modelo aprendió ya no sirve y el error casi se cuadruplica (de 1.18 a 4.29). **No lo hubiera detectado mirando solo los viajes que llegan:** un PSI sobre la distancia o la zona saldría como el de cualquier semana normal, porque las entradas no cambiaron. Solo lo muestra el error, al unir cada predicción con la tarifa cobrada.

## Ejercicio 1: ¿deriva de datos o deriva conceptual?

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`ejercicio1_deriva.ipynb`](./starter/ejercicio1/ejercicio1_deriva.ipynb) entrena la versión 1 del modelo de la tarifa y simula semanas de 300 viajes: semanas normales, semanas de feriado con más viajes al aeropuerto (deriva de datos) y semanas con un aumento de la tarifa por taxímetro (deriva conceptual). Para cada una mide el PSI de dos features contra los viajes de entrenamiento y el MAE del modelo. Los 4 `TODO`: la fórmula del PSI, medir 200 semanas normales, y las dos derivas simuladas.

Corre todas las celdas antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1 y 2):** antes de medir las semanas normales, ¿qué PSI esperas para una semana en la que nada cambió? Después de medirlas: ¿cuál fue el máximo en 200 semanas?

Antes de medir esperaba un PSI pequeño pero no 0, de unas centésimas (0.01 a 0.05): 300 viajes al azar nunca se reparten exactamente como los de entrenamiento. Después de medir: el promedio fue 0.050 en distancia y 0.035 en zona, y el **máximo en 200 semanas fue 0.1214 en distancia** (0.1091 en zona); el percentil 95, 0.092 y 0.080. Es decir, solo por azar una semana normal puede pasar de 0.10.

**Respuesta 2 (TODO 3 y 4):** antes de correr, ¿qué esperas que pase con el PSI y con el MAE en la semana de feriado? ¿Y en la semana del aumento?

En la semana de feriado esperaba que el PSI subiera mucho, sobre todo el de zona (más viajes desde JFK) y el de distancia (los viajes al aeropuerto son largos), y que el MAE cambiara poco, porque el modelo ya conoce esos viajes. En la del aumento esperaba lo contrario: el PSI igual que en una semana normal (los viajes son los mismos) y el MAE mucho más alto. Así fue: con peso 4, PSI de 0.189 en distancia y 0.196 en zona, y el MAE **bajó** a 0.97; con el aumento de 20%, PSI de 0.077 y 0.062 (normales) y el MAE subió de 1.17 a **4.16**.

**Respuesta 3 (Paso 4):** en la semana de feriado con peso 8, el PSI es más de diez veces el de una semana normal. ¿Empeoró el modelo? ¿Por qué?

No, mejoró: con peso 8 el PSI es 0.542 en distancia y 0.419 en zona (más de diez veces el 0.050 y 0.035 de una semana normal), pero el MAE bajó a 0.94, contra 1.19 en una semana sin cambios. Los viajes que sobran son los del aeropuerto, que tienen tarifa fija de 70: el modelo los aprendió bien (con `es_tarifa_fija`) y se equivoca muy poco en ellos, así que una semana con más de esos viajes tiene un error promedio menor. Un PSI alto dice que cambió lo que llega, no que el modelo esté peor: depende de si el modelo ya sabe resolver esos casos.

**Respuesta 4 (Paso 5):** ¿qué umbral de PSI elegirías para la alerta, y con qué dos números de la tabla lo justificas?

**0.15.** Con los números del Paso 5: con 0.15 hay **0% de falsas alarmas** en las 200 semanas normales (el máximo normal fue 0.121) y detecta el **97.5%** de las semanas de feriado con peso 4. Con 0.10 también detecta el 100%, pero da falsas alarmas el 4% de las semanas (unas dos al año), y con 0.20 la detección cae a 71.5%. 0.15 queda por encima de todo lo que hizo el azar y todavía ve casi todos los feriados.

**Lo que mostró la simulación:**

| Escenario | PSI de distancia | PSI de zona | MAE |
|---|---|---|---|
| Semanas normales (máximo de 200) | 0.1214 | 0.1091 | 1.3164 |
| Feriado, aeropuerto ×4 | 0.2211 | 0.1965 | 1.0441 |
| Aumento de 20% | 0.0456 | 0.0426 | 4.0505 |

El feriado y el aumento son el promedio de 10 semanas del Paso 4; con una sola semana (semilla 1): feriado 0.1887, 0.1964 y MAE 0.9694; aumento 0.0767, 0.0615 y MAE 4.1638. MAE en prueba: 1.1695.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que distingue las dos derivas por lo que cambia: la de datos cambia las entradas y se ve en el PSI, aunque no siempre empeore el modelo (el feriado lo mejoró); la conceptual cambia la relación entre entradas y objetivo, no mueve el PSI y solo se ve en el error, cuando llega la verdad (el aumento cuadruplicó el MAE con un PSI normal). Por eso una alerta de PSI sola no basta, ni para avisar de todo lo que empeora el modelo ni para decidir reentrenar.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó que un PSI alto no significa que el modelo esté peor, y que la deriva que más daño hizo en la simulación fue justo la que el PSI no ve.

## Ejercicio 2: la alerta, en el servicio

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de la tarifa de la sesión pasada tiene ahora una ruta `/deriva`. [`deriva.py`](./starter/ejercicio2/deriva.py) trae `MonitorDeDeriva`: guarda los últimos 300 viajes que recibió el servicio y calcula el PSI de `distancia_km` y de `zona_origen` contra la referencia que guarda `entrenar.py` (`referencia_deriva.json`), con el umbral de 0.15. Los pasos 1 a 3 tienen `TODO` (las proporciones numéricas, las categóricas y el PSI); los pasos 4 (guardar solo los últimos viajes) y 5 (el estado, con la alerta) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`. Después:

```
python entrenar.py
python simular_semanas.py
```

[`simular_semanas.py`](./starter/ejercicio2/simular_semanas.py) manda al servicio una semana normal, una de feriado y una del aumento, y muestra `/deriva` después de cada una (no lo modifiques).

**Respuesta 1:** ¿qué dijo `/deriva` después de cada semana? ¿Alguna alerta te sorprendió, o te faltó?

Después de la **semana normal**: PSI de 0.053 en distancia y 0.055 en zona, sin alerta. Después de la de **feriado**: 0.258 y 0.221, **con alerta** (los dos pasan de 0.15). Después de la del **aumento**: 0.058 y 0.048, sin alerta, porque la ventana de 300 ya tenía solo viajes de esa semana y son viajes normales. No me sorprendió ninguna; me **faltó** la del aumento, que es la semana que de verdad empeora el modelo, y el monitor no podía verla porque mira solo las entradas.

**Respuesta 2:** la alerta del monitor no dijo nada en la semana del aumento, que es la que más empeora el modelo. ¿Qué alerta de la sesión pasada sí la habría visto, y por qué tarda más?

La alerta del **MAE por semana** de la Semana 13 (`desempeno.py`: unir cada predicción con la tarifa cobrada por su `id_prediccion` y comparar el MAE con el de prueba). Habría visto el error casi cuadruplicado. Tarda más porque necesita la verdad: la tarifa cobrada llega al terminar cada viaje, y hace falta juntar suficientes viajes (una semana) para que el MAE no sea ruido. El PSI, en cambio, se calcula con cada viaje que entra, en el momento.

**Respuesta 3:** escribe la regla que usaría el equipo para decidir, con las dos alertas (la de deriva de este ejercicio y la del MAE de la sesión pasada), si lo que pasa es deriva de datos, deriva conceptual o nada. ¿Qué haría en cada caso?

La regla, cada semana:

- **PSI alerta y el MAE no:** deriva de datos que el modelo resuelve bien (como el feriado). Se revisa qué cambió (que no sea un error del pipeline, como un campo que llega vacío), se anota y se vigila; no se reentrena.
- **MAE alerta y el PSI no:** deriva conceptual (como el aumento). El mundo cambió y las entradas no lo muestran: se averigua la causa y se **reentrena con los viajes posteriores al cambio**; mientras tanto, se avisa que las tarifas estimadas salen bajas.
- **Las dos alertan:** llegan casos que el modelo no sabe resolver (deriva de datos que sí lo empeora, o las dos a la vez). Se reentrena con datos que incluyan esos casos, y se revisa si la referencia del PSI también hay que actualizarla.
- **Ninguna alerta:** nada; se sigue.

**Si terminas antes:** el monitor mira solo dos features. Inventa una semana con deriva de datos que este monitor no vería. ¿Empeoraría el modelo?

Una semana con la misma mezcla de zonas de origen y de distancias, pero con muchos más viajes **de madrugada** (por ejemplo, una semana de conciertos nocturnos) o con muchos más pagos en efectivo: el monitor solo mira `distancia_km` y `zona_origen`, así que no vería nada. Si empeoraría el modelo depende de si ya aprendió bien la hora: si tiene suficientes viajes nocturnos en entrenamiento, no; si en la noche hay pocos datos (o un recargo nocturno que el modelo aprendió mal), sí, y solo lo veríamos en el MAE. También pasaría desapercibido un cambio en `zona_destino`, que el monitor no vigila.

## Ejercicio 3: la deriva de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio instrumentado de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: la simulación, en `monitoreo/`

Un notebook o un script que, con tus datos de prueba:

- mida el PSI de al menos dos de tus features en ventanas normales (del tamaño de lo que recibe tu servicio en un período), para saber cuánto se mueve sin que nada cambie;
- simule una deriva de datos y una deriva conceptual que tengan sentido en tu problema, y mida el PSI y tu métrica de la Semana 11 en cada una;
- elija el umbral de la alerta con esos números, y guarde la referencia que necesita el servicio.

### Entregable 2: la alerta, en el servicio

Un monitor de deriva en tu servicio (puedes adaptar `deriva.py`) con una ruta que diga el PSI de la ventana actual y si hay alerta, y una prueba en `servicio/test_api.py` que compruebe que la alerta se enciende con una ventana alejada de la referencia.

### Entregable 3: `docs/monitoreo.md`

Agrega la sección **Deriva**: la tabla de la simulación, el umbral elegido y por qué, y qué deriva no detecta tu alerta.

**En tu proyecto, ¿qué deriva de datos es la más probable, y empeoraría tu modelo? ¿Y qué deriva conceptual?**

La deriva de datos más probable es un **cambio en el nivel de la demanda**: la temporada de lluvias o las vacaciones de diciembre (menos viajes) o un evento o una ciclovía nueva (más). En la simulación ([`docs/monitoreo.md`](../../psets/proyecto/docs/monitoreo.md), sección Deriva), la bajada **no** empeora el modelo (con la demanda al 50%, MAE 0.73 contra 1.03) pero la subida sí (al 150%, MAE 1.57 y costo asimétrico 3.47, sobre su umbral de 2.30). La deriva conceptual más probable es que, con la misma historia reciente, salgan más (o menos) bicicletas: una promoción, una estación vecina cerrada, o el propio rebalanceo: si el camión empieza a llegar a tiempo gracias al modelo, las estaciones ya no se vacían y las salidas que antes estaban limitadas por falta de bicis aumentan. Esa sí empeora el modelo (salidas ×1.5: MAE 1.57) y el PSI no la ve (0.03, como un día normal).

**Si tu alerta de deriva se enciende y tu métrica de desempeño no cambia, ¿qué haces?**

Primero, comprobar que la deriva es real y no un error del pipeline (por ejemplo, rezagos o llegadas que empiezan a llegar en 0 o vacíos por un cambio en el lote): eso sí hay que arreglarlo aunque el MAE no se haya movido todavía. Si es real, ver qué cambió (en mi proyecto, lo más común es un día de menos demanda, como un sábado o un día de lluvia), anotarlo y seguir vigilando el MAE y el costo los días siguientes. No reentreno solo por el PSI: si el modelo resuelve bien los casos nuevos, reentrenar no gana nada. Si la deriva se queda (una nueva normalidad), actualizo la referencia del PSI con datos recientes para que la alerta no se quede encendida y deje de significar algo.

## Antes de la próxima sesión

**Lectura:** ["How to Effectively Communicate Results to Stakeholders As a Data Scientist"](https://medium.com/@sahin.samia/how-to-effectively-communicate-results-to-stakeholders-as-a-data-scientist-711e88dd464f), Sahin Samia.

En las últimas cuatro sesiones construiste un servicio, un registro, métricas y alertas, con PSI, MAE, p95 y recall. ¿Qué cambia entre explicarle tu proyecto a otro científico de datos y a quien paga la factura? Escribe la primera oración que le dirías a cada uno.

Con otro científico de datos puedo hablar del método y de la evidencia: qué modelo, contra qué línea base, con qué validación y cuánta incertidumbre. A quien paga le importa la decisión: qué gana la operación, cuánto cuesta y qué tiene que aprobar, en su unidad (estaciones vacías, viajes perdidos, dinero), sin nombres de algoritmos ni de métricas. **Al científico de datos:** "Un gradient boosting predice las salidas de cada estación en los próximos 30 minutos con un MAE de 1.03, 16% menos que la persistencia, y lo validé con validación cruzada temporal." **A quien paga:** "Con el pronóstico, las veces que una estación muy usada se queda sin bicicletas sin que el camión lo vea venir bajan 23%, con el mismo camión y el mismo personal."
