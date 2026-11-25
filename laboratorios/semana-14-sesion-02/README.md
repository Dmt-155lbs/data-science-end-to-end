# Semana 14, Sesión 2: Comunicación técnica y ejecutiva de los resultados

> Objetivo de la sesión: adaptar la presentación de los resultados y limitaciones de tu proyecto a una audiencia técnica y a una audiencia ejecutiva.

**Lectura previa de esta sesión:** ["How to Effectively Communicate Results to Stakeholders As a Data Scientist"](https://medium.com/@sahin.samia/how-to-effectively-communicate-results-to-stakeholders-as-a-data-scientist-711e88dd464f), Sahin Samia.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 11: The Human Side of Machine Learning.

## Requisitos antes de esta sesión

Nada nuevo que instalar. Hoy se escribe más de lo que se programa: los dos scripts preparan los números y los gráficos, y el trabajo es decidir qué se dice a quién.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué hay que saber de la audiencia antes de preparar la presentación? ¿Qué recomienda hacer con la jerga técnica?**

Su **formación técnica y sus intereses**, porque cada audiencia necesita otra cosa: a los ejecutivos les importa el impacto en el negocio, el retorno de la inversión y qué pueden hacer con los resultados; a los equipos de producto o marketing, el comportamiento de los clientes y el desempeño del producto; a los equipos técnicos, la metodología, el desempeño del modelo y las dificultades técnicas. Con la jerga, recomienda **evitarla**: explicar el método "sin jerga", mantener los detalles técnicos claros pero simples, y usar metáforas o analogías cuando hagan falta. También traducir el desempeño del modelo (accuracy, precision, recall) a lo que significa para el negocio, y dejar la profundidad técnica para un apéndice aparte, para quien la pida.

**¿Qué lugar le da el artículo a las limitaciones y a la incertidumbre de los resultados?**

Lo pone dentro de **ser transparente**, que según el artículo es lo que genera confianza: hay que decir de entrada los supuestos y las limitaciones del análisis, y qué tan seguro se está de las predicciones o de las conclusiones. Además, recomienda llevar preparada la respuesta a "¿qué tan seguros están?" (qué tan confiables son los resultados y el modelo), presentar escenarios (el mejor y el peor caso) para que se vea el rango de resultados posibles según lo que decidan, y dejar los supuestos y las fuentes de datos en el apéndice técnico.

## Pregunta de apertura

¿Qué cambia entre explicarle tu proyecto a otro científico de datos y a quien paga la factura? Escribe la primera oración que le dirías a cada uno.

"Un gradient boosting predice las salidas de cada estación en los próximos 30 minutos con un MAE de 1.03, 16% menos que la persistencia, validado con validación cruzada temporal, y lo monitoreo con MAE diario y PSI."

"Con el pronóstico, las veces que una estación muy usada se queda sin bicicletas sin que el camión lo vea venir bajan 23%, con el mismo camión y el mismo personal: proponemos probarlo cuatro semanas."

## Ejercicio 1: noviembre, para dos audiencias

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`tarifas_noviembre.csv`](./starter/ejercicio1/tarifas_noviembre.csv) tiene los 1156 viajes de noviembre del monitoreo de la Semana 13: la tarifa estimada por la versión 1, la cobrada y si el viaje tenía tarifa fija. [`graficos_tarifa.py`](./starter/ejercicio1/graficos_tarifa.py) calcula el resumen por semana y dibuja dos gráficos con los mismos datos, `grafico_tecnico.png` y `grafico_ejecutivo.png`. Los 3 `TODO`: el resumen, el gráfico técnico y el gráfico ejecutivo.

```
python graficos_tarifa.py
```

**Respuesta 1 (TODO 1):** el MAE pasó de 1.18 a 4.29. Antes de calcularlo: ¿qué número de la tabla le dirías a la gerente de operaciones, que nunca oyó hablar del MAE?

El porcentaje de pasajeros que pagan más de 20% sobre la tarifa estimada (`pagan_20_mas`): pasó de **3% y 6%** en las dos primeras semanas a **47% y 46%** desde el 16 de noviembre. Casi la mitad de los pasajeros se lleva una sorpresa en el precio; eso lo entiende cualquiera. Lo acompañaría con la diferencia en dólares: desde el 16 de noviembre pagan en promedio **4.23 dólares más** de lo que les mostramos (antes, 0.16 y -0.07). El MAE (1.18 → 4.29) dice cuánto se equivoca el modelo, pero no en qué dirección ni a cuántas personas afecta.

**Respuesta 2 (TODO 2 y 3):** los dos gráficos usan los mismos viajes. Haz una lista de todo lo que cambia entre uno y otro: la métrica, el título, la cantidad de series, los colores.

| | Técnico | Ejecutivo |
|---|---|---|
| Métrica | MAE en dólares (error absoluto, sin dirección) | Porcentaje de pasajeros que pagaron más de 20% sobre lo estimado |
| Título | Describe qué se mide: "Error absoluto medio de la versión 1 por semana y tipo de tarifa" | Dice la conclusión: "Desde el 16 de noviembre, casi la mitad de los pasajeros paga más de 20% sobre la tarifa que les estimamos" |
| Series | Dos (taxímetro y tarifa fija), barras agrupadas, más la línea de referencia del MAE de prueba (1.17) y su leyenda | Una sola serie, sin leyenda ni referencia |
| Colores | Uno por serie (azul el taxímetro, gris la tarifa fija) | Gris las semanas normales y rojo las dos últimas: el color señala el problema |
| Valores | Se leen en el eje | Escritos encima de cada barra (3%, 6%, 47%, 46%); sin bordes de más |
| Ejes | Con la unidad de la métrica ("MAE (dólares)") | "Pasajeros que pagaron 20% más (%)" |

El técnico deja ver algo que el ejecutivo no: el error subió solo en los viajes con taxímetro (de 1.30 y 1.41 a 4.75 y 4.62) y no en los de tarifa fija (0.22 a 0.44), lo que apunta a un cambio en la tarifa por taxímetro.

**Respuesta 3:** escribe las dos diapositivas. Cada una tiene un título que es una oración completa, el gráfico y como máximo cuatro líneas.

**Diapositiva técnica** (para el equipo de datos):

**Desde el 16 de noviembre el MAE de la versión 1 casi se cuadruplicó (1.18 → 4.29), solo en los viajes con taxímetro: es deriva conceptual**

- Qué se midió: el MAE semanal de los 1156 viajes de noviembre, uniendo cada predicción con la tarifa cobrada por su `id_prediccion`, separado por tipo de tarifa, contra el MAE de prueba (1.17).
- Cómo cambió: taxímetro de 1.30 y 1.41 a 4.75 y 4.62; tarifa fija estable (0.22 a 0.44). La diferencia media pasó de +0.16 a **+4.23 dólares**: el modelo subestima, y las entradas (zonas, distancias) no cambiaron.
- Qué se concluye: cambió la relación entre el viaje y la tarifa por taxímetro (compatible con un aumento de tarifa); hay que reentrenar con viajes desde el 16 de noviembre.
- Limitación: no conocemos la causa ni el tamaño exacto del cambio; son 239 a 318 viajes por semana, y la semana del 23 está incompleta.

**Diapositiva ejecutiva** (para la gerente de operaciones):

**Desde el 16 de noviembre, casi la mitad de los pasajeros paga más de 20% sobre la tarifa que les mostramos**

- Qué pasó: desde el 16 de noviembre el cálculo de la tarifa no refleja lo que se cobra en los viajes con taxímetro; los del aeropuerto siguen bien.
- Qué cuesta: 47% de los pasajeros paga más de 20% sobre lo que vio en la app, y en promedio cada viaje cuesta 4.23 dólares más de lo anunciado: quejas y pérdida de confianza en el precio.
- Qué recomendamos: actualizar el cálculo con los viajes desde el 16 de noviembre (unos días de trabajo) y, mientras tanto, avisar en la app que la tarifa puede ser mayor.
- Qué necesitamos de usted: confirmar si cambió la tarifa oficial y desde cuándo, y aprobar el aviso en la app.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que adapta el mensaje a la decisión de cada audiencia y no solo el vocabulario: al equipo de datos le sirve la métrica, cómo se midió y sus limitaciones para diagnosticar; a la gerente, la conclusión en su unidad (pasajeros, dólares) y qué tiene que hacer. Los mismos datos, otra pregunta: el técnico responde "qué falla y por qué", el ejecutivo "qué pasa, cuánto cuesta y qué decido".

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó que traducir la métrica no basta: el mensaje ejecutivo tiene que terminar en una recomendación y en lo que se necesita de quien decide.

## Ejercicio 2: el fraude, en dólares

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El reporte técnico de la Semana 13 decía que el recall de la semana del 16 cayó de 0.74 a 0.28. [`costos_fraude.py`](./starter/ejercicio2/costos_fraude.py) lo traduce a dólares con los costos de la Semana 11 (40 por fraude no detectado, 3 por revisar un pago legítimo). Los pasos 1 a 3 tienen `TODO`: los errores por semana, su costo y el costo de una semana normal. Los pasos 4 (la oración para la gerencia) y 5 (lo que cuesta un mes así) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python costos_fraude.py`.

**Respuesta 1:** ¿qué oración te dio el paso 4? ¿Qué le falta para que la gerente pueda decidir algo?

"La semana del 2026-11-16, el fraude que no detectamos y las revisiones costaron 2359 dolares, 2.7 veces una semana normal." (la normal costó 867). Dice qué pasó y cuánto costó, pero no deja decidir nada: no dice por qué pasó (el modelo dejó pasar 55 de 76 fraudes), qué pasa si no se hace nada (un mes así cuesta unos **5,968 dólares más** que uno normal), qué recomendamos, qué necesitamos de ella ni qué tan seguro es el número. Además la fecha y el número están en formato de máquina ("2026-11-16", "2359").

**Respuesta 2:** escribe el mensaje ejecutivo completo, en cinco oraciones como máximo: qué pasó, cuánto cuesta, qué recomiendan, qué necesitan y qué riesgo queda.

La semana del 16 de noviembre, el sistema de fraude dejó pasar 55 de los 76 fraudes, y entre lo que se perdió y las revisiones costó 2,359 dólares, 2.7 veces una semana normal (867). Si sigue así, cada mes nos cuesta unos 6,000 dólares más que uno normal. Recomendamos actualizar el modelo con los fraudes confirmados de noviembre y, mientras tanto, revisar a mano más pagos de los que hoy marcamos. Necesitamos su aprobación para el costo extra de esas revisiones durante unas dos semanas y para publicar la versión nueva. El riesgo que queda: las semanas posteriores al 16 todavía no tienen todos sus fraudes confirmados, así que el costo real puede ser mayor, y la mejora solo se podrá confirmar unas tres semanas después del cambio.

**Respuesta 3:** el costo de 40 dólares por fraude no detectado es una suposición de la Semana 11. ¿Dónde lo dirías, en el mensaje ejecutivo o en el técnico? ¿Qué pasa con tu recomendación si el costo real fuera la mitad?

En el **técnico**, con los demás supuestos (de dónde sale, qué incluye y cómo cambia el resultado); en el ejecutivo como mucho una mención de que es un costo estimado, sin el detalle. Si el costo real fuera la mitad (20 dólares): la semana del 16 costaría 55 × 20 + 159 = **1,259 dólares**, contra 507 de una semana normal (487 y 527): **2.5 veces**, y un mes así, unos **3,008 dólares más** en vez de 5,968. La recomendación no cambia: el aumento relativo es casi el mismo, porque lo que lo mueve son los 55 fraudes no detectados (contra 17 y 19), y actualizar el modelo cuesta mucho menos que eso. Cambia la urgencia del refuerzo de revisiones manuales, que vale menos la pena si cada fraude cuesta menos.

## Ejercicio 3: los resultados de tu proyecto, dos veces

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/resultados_tecnico.pdf`

Una diapositiva para otra persona del equipo de datos: el modelo y la línea base, la métrica de la Semana 11 con su esquema de validación, lo que mostró el monitoreo y las limitaciones, con los números.

### Entregable 2: `docs/resultados_ejecutivo.pdf`

La misma diapositiva para quien decide si se usa el modelo: un título que dice la conclusión, el resultado en la unidad de quien decide (dinero, tiempo, casos), qué recomiendan, qué necesitan y qué no hace el modelo. Sin el nombre del algoritmo ni de la métrica.

Puedes armar las dos diapositivas en lo que prefieras (Google Slides, PowerPoint, Markdown exportado) y guardarlas en PDF.

**¿Qué número de tu proyecto usaste en la versión ejecutiva que no aparece en la técnica? ¿De dónde salió?**

Los **momentos de mucha demanda en que la estación se quedó corta por 2 bicicletas o más**: 7,926 en la semana de prueba con la regla actual (lo que salió en la última media hora) contra 6,113 con el pronóstico, **1,813 menos (23%)**, unas 259 por día. Lo calculé en [`docs/diapositivas_resultados.py`](../../psets/proyecto/docs/diapositivas_resultados.py) con el registro de la semana simulada (`monitoreo/predicciones.jsonl`, 67,100 predicciones unidas con sus salidas reales): momentos de mucha demanda son las franjas de estación con 3 salidas o más en la media hora siguiente (15,625 en la semana), y cuento en cuántas la predicción quedó 2 o más por debajo. La versión técnica habla de MAE y costo asimétrico; la ejecutiva, de veces que una estación muy usada se queda sin aviso.

**¿Qué limitación dejaste en las dos versiones, y cómo la dijiste en cada una?**

Que está probado con **un solo mes de datos** (agosto de 2026) y 100 estaciones. En la técnica: "Limitaciones: solo un mes y 100 estaciones; se queda corto en 39% de las franjas (apunta al centro); el PSI no ve deriva conceptual; el costo 3:1 es un supuesto", con los números. En la ejecutiva, en el bloque de "Qué no hace": "solo está probado con un mes (agosto) y se revisa si cambia la demanda", junto con "todavía falla en 4 de cada 10 momentos de mucha demanda"; por eso la recomendación es un piloto de 4 semanas y no ponerlo en producción de una vez.

## Antes de la próxima sesión

**Lectura:** ["AI Model Lifecycle Management: From Development to Decommissioning"](https://blog.cognitiveview.com/ai-model-lifecycle-management-from-development-to-decommissioning/), CognitiveView.

Hoy escribiste "qué no hace el modelo" en una diapositiva. Ahora imagina que sí lo hace mal: ¿bajo qué condición apagarías el modelo de tu proyecto, y quién tendría que enterarse?

Lo apagaría (y volvería a la regla actual, la persistencia, que no necesita modelo) si durante **dos semanas seguidas** su costo asimétrico queda **por encima del de la persistencia** en los mismos días (2.35 en la validación) o su MAE por encima de 1.23: a partir de ahí el modelo hace peor lo mismo que la regla simple, y reentrenarlo no lo arregló. También lo apagaría de inmediato si los datos de entrada dejan de ser confiables (por ejemplo, si el feed GBFS cambia de formato o el lote empieza a mandar rezagos vacíos) y no se puede corregir el mismo día. Tendrían que enterarse los despachadores y el jefe de operaciones de rebalanceo (cambia cómo ordenan las rutas), quien paga el proyecto en Ecobici y el equipo de datos; y, como dice la lectura, documentar por qué se retiró y guardar el registro de predicciones y la versión del modelo para poder auditarlo después.
