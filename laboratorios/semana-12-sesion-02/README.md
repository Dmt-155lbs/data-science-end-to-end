# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

El artículo sigue estos pasos: entrenar el modelo y **guardarlo en un archivo** (con `joblib.dump`, `lin_regress.sav`); escribir una **API REST con FastAPI** que define el formato de la petición con un `BaseModel` de pydantic, carga el modelo una vez al arrancar y expone una ruta `GET /` y una `POST /predict` que recibe los datos nuevos y devuelve la predicción en JSON; **levantarla con uvicorn**, indicando el host y el puerto (`0.0.0.0`, 8000), y probarla con `requests`; y, por último, **empaquetarla con Docker**: un `Dockerfile` que instala Python y las dependencias de `requirements.txt`, copia el proyecto, expone el puerto 8000 y corre uvicorn; con él se construye una imagen (`docker build . -t apiserver`) y se corre el contenedor (`docker run -p 8000:8000 apiserver`). Su idea de fondo: "un modelo en un notebook vale 0 dólares" hasta que otra aplicación lo puede usar.

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

Para que el servicio corra igual en cualquier otra máquina, no solo en la del autor. Su ejemplo: un equipo de 20 personas que quiere tener la API corriendo; instalar todo en cada computadora sería un dolor de cabeza por los distintos sistemas operativos, dependencias y versiones. El contenedor empaqueta la aplicación con todas sus dependencias, así que basta tener Docker y la imagen para correrla sin preocuparse de instalar las versiones correctas. Además es más rápido y liviano que un entorno virtual o una máquina virtual.

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un servicio de predicción en línea: un programa que corre todo el tiempo con el modelo ya cargado en memoria, detrás de una API (una ruta `POST /predecir`), y que responde en menos de un segundo. La app le mandaría, en un JSON, lo que sabe del viaje en ese momento: la fecha y hora del pedido, la zona de origen y la de destino, la distancia estimada por la ruta, los pasajeros y el método de pago. El servicio calcula el resto de las features (hora como ciclo, fin de semana, hora pico, tarifa fija) igual que se calcularon para entrenar, y responde la tarifa estimada.

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que todo dice que funciona y nada funciona: `/predecir` responde 200 con una tarifa de 0.0 para cualquier viaje, porque el código todavía no llama al modelo. `/salud` solo dice que el archivo del modelo se cargó, no que se use, y `pytest` da `3 passed` porque las pruebas de los `TODO 3` y `4` no tienen `assert`. Un código 200 y una marca verde no dicen que la respuesta sea correcta: hace falta una prueba que compruebe el valor.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

Al viaje sin `distancia_km`, una tarifa: en el esquema es opcional (`None` por defecto) y la preparación la imputa con la mediana de entrenamiento. Así fue: el viaje de Midtown al aeropuerto sin distancia dio 68.56, contra 70.39 con ella. Al viaje sin `zona_origen`, un error 422: la zona es obligatoria en el esquema, FastAPI lo rechaza antes de entrar a la función y responde qué campo falta.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El modelo se equivocaría sin dar ningún error. Aprendió cuánto suben las tarifas cuando `es_hora_pico` vale 1, con las horas 7 a 9 y 16 a 19 entre semana; si el servicio marcara otras horas, le aplicaría ese aumento a viajes que no lo tienen (y se lo quitaría a los que sí). La respuesta sigue siendo un número con código 200, así que nadie lo nota. Por eso el servicio importa `agregar_features` de `features_taxis.py` en vez de reescribirla: es la misma función con que se construyeron las features de entrenamiento.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce FastAPI, no mi código: valida el JSON contra el esquema `Viaje` antes de llamar a la función `predecir`. Si falta un campo obligatorio, responde 422 con el detalle (`"loc": ["body", "zona_origen"]`, `"msg": "Field required"`) y la petición no llega ni a `agregar_features` ni al modelo.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Porque uvicorn, por defecto, escucha solo en `127.0.0.1`, la dirección interna del propio contenedor. El log lo dice (`Uvicorn running on http://127.0.0.1:8000`): está corriendo, pero solo acepta conexiones que nazcan dentro del contenedor. `docker run -p 8000:8000` reenvía el puerto desde afuera hacia la interfaz de red del contenedor, no hacia su `127.0.0.1`, así que el `curl` de afuera no encuentra a nadie (`curl` falló con código 56). Con `--host 0.0.0.0` uvicorn escucha en todas las interfaces y el mismo `curl` responde `{"estado":"ok","modelo_cargado":true}`.

**Lo que respondió el servicio:**

- **Viaje de Midtown a JFK Airport** (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta): `{"tarifa_estimada": 70.39}`, la tarifa fija de 70 del aeropuerto. Igual desde uvicorn y desde el contenedor.
- **El mismo viaje sin zona de origen**, código 422:

```
{"detail":[{"type":"missing","loc":["body","zona_origen"],"msg":"Field required","input":{"fecha_hora_inicio":"2026-11-05 08:30","zona_destino":"JFK Airport","distancia_km":26.4,"pasajeros":2,"metodo_pago":"tarjeta"}}]}
```

`pytest -v`: 3 passed.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que separa el contrato del servicio del modelo: el esquema de entrada decide qué datos se aceptan y rechaza lo que no cumple (422) antes de llegar al modelo, el servicio calcula las features con la misma función del entrenamiento para no cambiar en silencio lo que el modelo ve, y el contenedor empaqueta el servicio con las mismas versiones de las librerías para que corra igual en cualquier máquina. Una respuesta con código 200 no garantiza que la predicción sea correcta; eso lo garantizan las pruebas que comprueban el valor.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó que los errores peligrosos de un servicio son los silenciosos (una feature calculada distinto, una tarifa en 0 con código 200), no los que dan 422, y que por eso hacen falta pruebas con `assert` sobre lo que responde.

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

- **De madrugada** (02:40, app, 41.30 dólares, 8.6 km, tarjeta nueva, 2 intentos fallidos, 0 viajes previos): `{"probabilidad_fraude": 0.9996, "revisar": true}`.
- **De la tarde** (16:15, tarjeta, 18.40 dólares, 6.2 km, tarjeta conocida, 0 intentos fallidos, 22 viajes previos): `{"probabilidad_fraude": 0.0165, "revisar": false}`.

`comprobar.py`: los cinco pasos OK.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app no tiene que saber nada del modelo ni del umbral: recibe una decisión y actúa (mandar el pago a revisión o aprobarlo). Si cada app aplicara su propio umbral, podrían quedar distintos entre sí, y cambiarlo obligaría a actualizar todas. Si mañana el umbral es 0.20, se cambia en un solo lugar, la constante `UMBRAL` de `servicio_fraude.py`, y se vuelve a desplegar el servicio: la app no cambia. (Si hay pruebas que fijan la decisión con el umbral, como la del pago de 0.25, hay que revisarlas: 0.25 sigue pasando 0.20.)

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Sí: el pago de madrugada da 0.9996 en este servicio y 0.9996 en el lote de la sesión pasada. Para que así sea, tienen que ser iguales tres cosas: el modelo (el mismo archivo `.joblib`, entrenado con los mismos datos y la misma versión de scikit-learn), las features (las dos usan `agregar_features` y `columnas` de `preparacion_pagos.py`: la misma hora y el mismo `monto_por_km`) y los datos del pago. Si cualquiera de los dos calculara una feature a su manera, el mismo pago daría dos probabilidades distintas.

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde 422 sin llegar al modelo: `{"detail":[{"type":"float_parsing","loc":["body","monto"],"msg":"Input should be a valid number, unable to parse string as a number","input":"cuarenta"}]}`. La produce la línea `monto: float` del esquema `Pago`: FastAPI (con pydantic) valida el tipo de cada campo antes de llamar a `evaluar_pago`, y "cuarenta" no se puede convertir en número.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

Recibe una franja de 15 minutos de una estación tal como la conoce quien llama (el lote de cada 15 minutos): el código de la estación, la fecha, la hora, la capacidad, las salidas y llegadas de la franja que acaba de cerrar y los rezagos (salidas de la franja anterior, de la última hora y del mismo momento de ayer), que pueden llegar vacíos. El servicio calcula `dia_semana` y `es_fin_de_semana` a partir de la fecha, y `periodo_demanda` y `salidas_hora_por_capacidad`, con `agregar_calendario` y `agregar_features` importadas de `modelado/pipeline_proyecto.py`. Sé que las calcula igual por tres comprobaciones: `agregar_features` es la misma función que usa `cargar_y_dividir` para entrenar; la prueba `test_calendario_igual_que_en_entrenamiento` confirma que el día de la semana sigue la convención del archivo de entrenamiento (lunes 1, sábado 6, y en todo agosto coincide); y la estación 495 a las 8:00 del 26 de agosto da **10.5** tanto por el servicio como por el camino del entrenamiento (`cargar_y_dividir` y el mismo modelo). Pruebas en [`servicio/test_api.py`](../../psets/proyecto/servicio/test_api.py): `pytest psets/proyecto` da 9 passed.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

Con la imagen construida desde `psets/proyecto/` (`docker build -f servicio/Dockerfile -t servicio-ecobici .`) y corriendo con `docker run -p 8020:8000 servicio-ecobici`:

```
curl -X POST localhost:8020/predecir -H "Content-Type: application/json" -d '{"estacion_codigo": "495", "fecha": "2026-08-26", "hora": 8, "capacidad": 39, "salidas": 4, "llegadas": 6, "salidas_lag_15m": 10.0, "salidas_ultima_hora": 40.0, "salidas_mismo_momento_ayer": 9.0}'
```

```
{"estacion_codigo":"495","salidas_prox_30m":10.5,"run_id":"7749f7c3a8a74957baf3eb06d655ba7b"}
```

En realidad salieron 7 bicis en esos 30 minutos. Es una de las franjas con más demanda del periodo de prueba (40 salidas en la última hora), donde el modelo se pasa.

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Con un archivo que se reemplaza no lo sabría: el `modelo.joblib` no dice de qué corrida salió, y el de hoy desaparece cuando exporto el de la semana que viene. Lo que hace falta es lo que la lectura de IBM llama control de versiones e inventario de modelos: cada modelo registrado con su versión, sus datos y su responsable. En la práctica, que cada predicción guarde qué versión la produjo (mi servicio ya devuelve el `run_id` en cada respuesta, igual que la `version_modelo` del lote de la Semana 12), y que el servicio cargue una versión registrada en MLflow en vez de un archivo suelto. Así, para el reclamo de mañana, busco la versión en el registro de esa predicción; y para volver al modelo anterior, apunto el servicio a la versión anterior del registro, que sigue guardada, sin reentrenar nada.
