# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

El aprendizaje clásico conviene con datos tabulares y estructurados, una etiqueta bien definida y un conjunto modesto, sobre todo si importan la latencia o la interpretabilidad. Un LLM conviene cuando la entrada es texto, la tarea exige razonar sobre contexto o no hay datos etiquetados.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

Los datos. Mi proyecto usa filas numéricas de estaciones con una etiqueta clara, que es justo el caso que el artículo asigna al aprendizaje clásico; eso pesa más que el costo o la explicabilidad.

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Costo por predicción, tiempo de respuesta, si el mismo viaje da siempre la misma tarifa, si se puede explicar la predicción y cuánto cuesta entrenar y mantener cada opción.

## Ejercicio 1: una red neuronal frente al ensamblaje

Archivo: [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Significa que la red llegó al límite de iteraciones antes de que el entrenamiento se estabilizara, así que quedó sin terminar de aprender. Con 1000 iteraciones espero que el aviso desaparezca y el error baje. Bajó a 1.1695 en prueba y superó al ensamblaje.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

1729: 25 × 64 = 1600 pesos de entrada, 64 sesgos de la capa oculta, 64 pesos hacia la salida y 1 sesgo de salida.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

La distancia, y lo confirmó con 0.9824 de importancia. Para explicar los 70 dólares sirve más el ensamblaje: `es_tarifa_fija` es su segunda columna y el árbol deja ver esa decisión, mientras que la red son 1729 números sin una lectura directa. Las importancias son globales, así que explicar un viaje puntual exige además seguir sus árboles.

**Lo que confirmó la salida, TODO por TODO:**

| Modelo | Entrenamiento | Prueba | Segundos |
|---|---|---|---|
| Gradient boosting | 1.0705 | 1.2754 | 0.2 |
| Red neuronal | 1.0882 | 1.1695 | Entre 4 y 9 |

Parámetros de la red: 1729. Columnas más importantes del ensamblaje: `distancia_km` (0.9824), `es_tarifa_fija` (0.0120) y `es_hora_pico` (0.0028).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que dice que la decisión no se toma solo por el error. Con errores parecidos (red neuronal 1.1695 y gradient boosting 1.2754 en prueba), pesan el costo de cada predicción, el tiempo de entrenamiento, que el mismo viaje dé siempre la misma tarifa y poder explicarla. El ensamblaje entrena en 0.2 s y deja ver que `distancia_km` y `es_tarifa_fija` mandan; la red son 1729 parámetros sin lectura directa, y un LLM cobra por cada consulta.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión reforzó que, en datos tabulares con una etiqueta clara, el modelo clásico suele ser la mejor opción por costo y explicabilidad, y que un LLM tiene sentido cuando la entrada es texto o no hay datos etiquetados.

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Archivo: [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

Usé Claude en un chat nuevo, sin contexto. El costo es con 2 dólares por millón de tokens de entrada.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.8957 | 0.19 s | Casi 0 | Sí, con importancias y árboles |
| Red neuronal | 0.9637 | 4.42 s | Casi 0 | Difícil: 1729 parámetros |
| LLM (Claude) | 1.1395 | 0 | 0.73 dólares (Sonnet 5.5, solo entrada) | No de forma comprobable |

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

No. En una conversación nueva (Claude Sonnet, sin contexto) solo 1 de los 20 números coincidió, y el error pasó de 1.1395 a 2.7335. El peor caso fue un viaje de JFK a Midtown de 37.29 km: la primera vez respondió la tarifa fija de 70 y la segunda 95, como si la tarifa dependiera de la distancia. Para una app que cobra, eso significa que el mismo viaje puede costar distinto según cuándo se pregunte, y que ni siquiera la regla fija está garantizada.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

Fue un viaje de JFK a JFK de 1.01 km: el LLM dijo 7 dólares y el real era 11.46, un error de 4.46. Tiene poca distancia, pero sale y llega al aeropuerto. El gradient boosting, que entrenó con las 2380 filas, dio 12.74, un error de 1.28, así que este caso sí lo capturó mucho mejor que el LLM con 30 ejemplos.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

Porque habría que mandar los 595 viajes al LLM, con más costo y trabajo manual. Una comparación sobre 20 viajes es poco confiable: en los 20 ganó el ensamblaje (0.8957 contra 0.9637), y en los 595 ganó la red (1.1695 contra 1.2754). Con tan pocas filas, el orden puede cambiar por azar.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.py`

Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

_(tu tabla)_

_(tu decisión)_

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

_(tu respuesta)_
