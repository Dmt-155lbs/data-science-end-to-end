# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los laboratorios de hoy son notebooks de Jupyter, como el diagnóstico de la Semana 4: cada gráfico aparece debajo de la celda que lo dibuja. Una exploración se lee para decidir qué hacer con los datos; por eso va en un notebook y no en un script.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

Qué datos tengo y en qué estado están, antes de decidir cómo prepararlos y modelarlos. El artículo lo organiza en tres pasos: **una visión general** (cuántas observaciones y variables hay, de qué tipo, cuántos faltantes y cuántos duplicados: "¿con qué estoy trabajando?"); **el análisis de las variables**, una por una y en relación con las demás (sus distribuciones, valores fuera de rango, interacciones y correlaciones, incluida la relación con el objetivo para encontrar las variables más útiles y posibles fugas); y **la evaluación de la calidad** de los datos (faltantes, desbalance, valores constantes o casi todos en cero, duplicados, variables redundantes o muy correlacionadas, ruido). Lo que se encuentra ahí guía todo lo que sigue: qué filas quitar, qué variables transformar o descartar y cómo evaluar el modelo.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Para una sola variable (análisis univariado), estadísticas descriptivas y una gráfica según su tipo: en las numéricas, media, desviación, asimetría, curtosis y cuantiles, representadas con un **histograma**; en las categóricas, la moda y una **tabla de frecuencias**, representadas con un **gráfico de barras**. Para la relación entre dos (análisis multivariado), las **interacciones** (un gráfico que muestra cómo se mueven los valores de una variable con los de la otra, como una dispersión) y las **correlaciones**, que le ponen un número a la fuerza de esa relación y se leen en una matriz o en un **mapa de calor**. El artículo propone hacer todo eso de una vez con un reporte de **ydata-profiling**, que además genera alertas automáticas de calidad.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

El modelo aprende de esos 15 viajes como si fueran reales: para acercarse a tarifas de 999.99 sube sus estimaciones en general (en una regresión lineal, unos pocos valores extremos tiran de la recta), así que se equivoca más con los viajes normales, y el promedio de la tarifa ya no representa nada (31.39 con los errores, contra 26.52 sin ellos). Mirando solo el resultado final no lo notaría: vería un error alto y no sabría que viene de 15 filas mal capturadas, y podría culpar al modelo y no a los datos. Se descubre explorando antes de entrenar: el máximo de 999.99 en `describe()` y el histograma lo delatan de inmediato.

## Ejercicio 1: resumen, distribución y atípicos

Notebook: [`starter/ejercicio1_explorar.ipynb`](./starter/ejercicio1_explorar.ipynb), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York, con fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Ya lee el archivo y arma los gráficos; faltan 4 `TODO`. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

El **máximo de la tarifa, 999.99**, absurdo cuando el 75% de los viajes cuesta 32.55 o menos (y que, además, es un número "redondo" típico de un valor de relleno), y el **mínimo de la duración, -74.00 minutos**, imposible: un viaje no puede durar menos de cero. También llama la atención la desviación de la tarifa (70.99), mucho mayor que su media (31.39), y la distancia mínima de 0.00 km.

**Respuesta 2 (TODO 2):** mira el histograma después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

Porque la escala del eje x la fijan los 15 viajes de 999.99: con 50 barras que van de 1.60 a 999.99, cada barra mide unos 20 dólares y casi todos los viajes (que cuestan entre 14 y 33) caen en las primeras dos o tres barras, mientras el resto del eje queda vacío hasta el grupo de 999.99 a la derecha. Unos pocos valores extremos aplastan la distribución real contra el borde izquierdo.

**Respuesta 3 (TODO 3 y 4):** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No: espero que la mayoría no sean errores. La regla del IQR solo dice que un valor está lejos del centro, no que esté mal. Lo averiguaría mirando qué valores son los atípicos (`value_counts()`) y si tienen una explicación del dominio. Así fue: de los 334 viajes sobre el límite, **302 cuestan exactamente 70.00** (la tarifa fija entre JFK Airport y Manhattan, un precio real) y solo **15 son el error de captura de 999.99**; los demás son viajes largos y caros pero posibles. Los que hay que quitar son los 15 de 999.99 y los 10 de duración negativa; los de 70.00 se quedan, porque son información.

**Lo que confirmó la salida, TODO por TODO:**

- **Resumen (TODO 1):** 3000 viajes y 10 columnas; faltan 120 valores de `pasajeros` y 60 de `metodo_pago`. Tarifa: media 31.39, mediana 22.60, máximo 999.99. Duración: mínimo -74.00.
- **Histograma (TODO 2):** casi todos los viajes comprimidos en las primeras barras, con un grupo aislado en 999.99.
- **IQR (TODO 3):** Q1 13.99, Q3 32.55, IQR 18.56, **límite superior 60.40**. **334 viajes** sobre el límite.
- **Cómo se reparten (TODO 4):** **302** con tarifa exacta de 70.00, **15** con el error de captura de 999.99 y 17 viajes caros pero posibles (entre 61.07 y 77.81). Además, **10 viajes** con duración negativa.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que no trata todo atípico como un error: un valor lejos del centro puede ser un error de captura (999.99, una duración negativa), que se quita, o un comportamiento real del negocio (la tarifa fija de 70 al aeropuerto), que se conserva y que incluso conviene darle al modelo como información. Para distinguirlos hay que mirar los valores y su contexto, no solo aplicar la regla del IQR; borrar los 334 atípicos habría eliminado los 302 viajes de tarifa fija, que son reales.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó que la regla del IQR solo señala qué revisar, y que la decisión de quitar una fila sale del conocimiento del dominio, no de la regla.

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Notebook: [`starter/ejercicio2_relaciones.ipynb`](./starter/ejercicio2_relaciones.ipynb), sobre el mismo `viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0. La última celda imprime una comprobación por TODO: completa los 5 hasta que todas digan `OK`.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

Además de la tendencia general (más distancia, más tarifa: correlación de 0.96), el gráfico muestra una **línea horizontal en 70.00**: **300 viajes** que cuestan exactamente lo mismo sin importar la distancia (de 0 a 45 km), y todos tocan JFK Airport, en viajes entre el aeropuerto y Manhattan. Es la tarifa fija del aeropuerto. Qué cambia en el modelo: esos viajes no siguen la regla de precio por kilómetro, así que hace falta una feature que los marque (`es_tarifa_fija`, que se construye en la Semana 9); sin ella, distorsionan la pendiente de la distancia para todos los demás.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Los **888 viajes pagados en efectivo tienen propina 0 en todos los casos** (promedio 0.00), mientras que los 2027 con tarjeta tienen propina siempre (promedio 4.73, ninguno en 0). No describe a los pasajeros sino al **sistema que registra los datos**: es casi imposible que nadie deje propina en efectivo; lo que pasa es que la propina en efectivo no se registra. Qué cambia en el modelo: la propina en efectivo no es un dato real sino un faltante disfrazado de 0, así que no se puede usar para concluir que quien paga en efectivo no deja propina, y si alguna vez fuera objetivo, habría que modelar solo los pagos con tarjeta.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La duración tiene una correlación de 0.956 con la tarifa, casi igual que la distancia (0.960), pero solo se conoce cuando el viaje termina; al pedir el taxi todavía no existe. Usarla sería una **fuga de información**: el modelo se vería excelente en la evaluación y fallaría en producción, donde esa columna no llega. Qué cambia en el modelo: la duración se deja fuera de las entradas. La distancia sí se puede usar, porque la app la estima con la ruta; y como las dos están muy correlacionadas entre sí (0.951), la distancia ya trae casi toda la información de la duración, incluida la parte del tráfico que se puede anticipar con la hora (la idea de `es_hora_pico` de la Semana 9). Los pasajeros apenas se relacionan con la tarifa (-0.051).

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar. Copia la plantilla [`plantilla-eda/explorar_proyecto.ipynb`](./plantilla-eda/explorar_proyecto.ipynb) a `psets/proyecto/modelado/`: el notebook es tu entregable, con el código, lo que mostró y lo que concluyes en el mismo lugar.

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3: filas, columnas, faltantes por columna, la distribución del objetivo y cuántos atípicos marca el IQR.

### Paso 2: relaciones (10 minutos)

Completa el paso 4: la relación entre tu objetivo y la variable que esperas que más influya.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

En el paso 5 del notebook, para cada hallazgo: una celda de código con el número o el gráfico que lo confirma, y debajo, en la celda de texto, qué encontraste y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar). Antes de hacer commit, reinicia el kernel y corre todo.

**Ruta de tu notebook de exploración en tu fork:**

[`psets/proyecto/modelado/explorar_proyecto.ipynb`](../../psets/proyecto/modelado/explorar_proyecto.ipynb): la exploración del flujo de estaciones de Ecobici (agosto de 2026, 297,600 franjas), con sus tres hallazgos. Lo hice el 20 de octubre como script (`plantilla-eda/explorar_proyecto.py`) y lo pasé al formato de notebook cuando el profesor cambió la plantilla.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Construyendo columnas nuevas que hagan explícito cada patrón, porque el modelo solo ve columnas. Para la tarifa fija, una columna `es_tarifa_fija` que vale 1 cuando el viaje va de JFK Airport a Manhattan o al revés, para que el modelo les asigne su propio valor en vez de aplicarles el precio por kilómetro. Para la hora, derivarla de `fecha_hora_inicio`: la hora (como ciclo, con seno y coseno, para que las 23 y las 0 queden cerca), el día de la semana, si es fin de semana y una marca de hora pico, que existen al momento de pedir el taxi.
