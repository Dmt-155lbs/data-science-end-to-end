"""Las dos diapositivas de resultados del proyecto (Semana 14, Sesion 2):
docs/resultados_tecnico.pdf, para el equipo de datos, y
docs/resultados_ejecutivo.pdf, para quien decide si se usa el modelo.
Desde psets/proyecto/, despues de python -m monitoreo.simular:

    python docs/diapositivas_resultados.py

Los numeros salen de los CSV del modelado y del monitoreo; los de la version
ejecutiva (momentos de demanda alta en que la prediccion se quedo corta por 2
bicicletas o mas) se calculan aqui con el registro de la semana simulada.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = Path(__file__).resolve().parent
PROYECTO = AQUI.parent

import sys  # noqa: E402

sys.path.insert(0, str(PROYECTO))
from modelado.pipeline_proyecto import OBJETIVO, RUTA_DATOS  # noqa: E402

AZUL, GRIS, ROJO, TINTA = "#2F5D8A", "#B0B0B0", "#C44E52", "#222222"


def momentos_cortos():
    """Franjas de la semana simulada con 3 o mas salidas reales en que la
    prediccion se quedo corta por 2 o mas, con el modelo y con la regla actual
    (persistencia: los proximos 30 minutos iguales a los ultimos 30)."""
    registro = pd.read_json(PROYECTO / "monitoreo" / "predicciones.jsonl", lines=True, dtype={"estacion_codigo": str})
    registro["fecha"] = pd.to_datetime(registro["fecha"]).dt.strftime("%Y-%m-%d")
    verdad = pd.read_csv(RUTA_DATOS, dtype={"estacion_codigo": str}).dropna(subset=[OBJETIVO])
    unidas = registro.merge(verdad[["estacion_codigo", "fecha", "franja", OBJETIVO]], on=["estacion_codigo", "fecha", "franja"])
    real = unidas[OBJETIVO]
    persistencia = unidas["salidas"] + unidas["salidas_lag_15m"].fillna(0)
    alta = real >= 3
    return {
        "franjas_alta": int(alta.sum()),
        "modelo": int((alta & (real - unidas["salidas_prox_30m"] >= 2)).sum()),
        "regla_actual": int((alta & (real - persistencia >= 2)).sum()),
    }


def lienzo(titulo):
    figura = plt.figure(figsize=(13.33, 7.5))
    figura.text(0.04, 0.94, textwrap.fill(titulo, 80), fontsize=20, weight="bold", color=TINTA, va="top")
    return figura


def lineas(figura, textos, x=0.55, y=0.76, ancho=56, tamano=12.5):
    for texto in textos:
        envuelto = textwrap.fill(texto, ancho)
        figura.text(x, y, envuelto, fontsize=tamano, color=TINTA, va="top", linespacing=1.35)
        y -= 0.031 * (envuelto.count("\n") + 1) + 0.045


def pie(figura, texto):
    figura.text(0.04, 0.025, textwrap.fill(texto, 175), fontsize=9, color="#666666")


def tecnica(corridas, negocio, monitoreo_dia):
    figura = lienzo("El gradient boosting baja el MAE 16% frente a la persistencia y se sostuvo en una semana "
                    "simulada en producción, pero no ve la deriva conceptual")
    eje = figura.add_axes([0.06, 0.16, 0.44, 0.6])
    nombres = ["Persistencia\n(línea base)", "Ridge", "Gradient\nboosting", "Red neuronal"]
    valores = corridas["mae_prueba"].tolist()
    colores = [GRIS, GRIS, AZUL, GRIS]
    barras = eje.bar(nombres, valores, color=colores)
    for barra, valor in zip(barras, valores):
        eje.text(barra.get_x() + barra.get_width() / 2, valor + 0.015, f"{valor:.3f}", ha="center", fontsize=12)
    eje.set_ylabel("MAE en prueba (salidas por franja, 25 al 31 de agosto)", fontsize=11)
    eje.set_ylim(0, max(valores) * 1.2)
    eje.spines[["top", "right"]].set_visible(False)
    gb = negocio.set_index("modelo").loc["gradient boosting"]
    pe = negocio.set_index("modelo").loc["persistencia"]
    lineas(figura, [
        "Qué se midió: salidas de cada una de 100 estaciones en los próximos 30 min, por franja de 15 min "
        "(agosto 2026, 297,600 franjas). División por fecha: días 1-24 entrenamiento, 25-31 prueba.",
        f"Validación cruzada temporal (5 pliegues): MAE 1.000 ± 0.037, contra 0.993 ± 0.032 de la red: "
        f"empate dentro del ruido; elegí el más rápido y explicable. Costo asimétrico (subestimar ×3): "
        f"{gb['costo_asimetrico']:.2f} contra {pe['costo_asimetrico']:.2f} de la persistencia.",
        f"Monitoreo (semana simulada, 67,200 peticiones): MAE {monitoreo_dia['mae_semana']:.4f}, p95 de 51 ms, 0 errores. "
        "Alerta de deriva: PSI de 2 features por día, umbral 0.12 (peor día normal 0.099).",
        f"Limitaciones: solo un mes y 100 estaciones; se queda corto en {gb['subestimadas']:.0%} de las franjas "
        "(apunta al centro); el PSI no ve deriva conceptual (salidas ×1.5: PSI 0.03, MAE 1.57); "
        "el costo 3:1 es un supuesto.",
    ])
    pie(figura, "Fuentes: modelado/corridas_proyecto.csv, validacion_proyecto.csv, metricas_negocio_proyecto.csv; "
                "monitoreo/desempeno_por_dia.csv, deriva_simulacion.csv. Detalle en docs/decision_modelo.md y docs/monitoreo.md.")
    figura.savefig(AQUI / "resultados_tecnico.pdf")
    plt.close(figura)


def ejecutiva(cortos):
    menos = cortos["regla_actual"] - cortos["modelo"]
    figura = lienzo(f"El pronóstico reduce {menos / cortos['regla_actual']:.0%} las veces que una estación muy usada "
                    "se queda sin aviso: proponemos probarlo 4 semanas con un despachador")
    eje = figura.add_axes([0.06, 0.16, 0.42, 0.58])
    nombres = ["Como hoy\n(lo que salió en la\núltima media hora)", "Con el pronóstico"]
    valores = [cortos["regla_actual"], cortos["modelo"]]
    barras = eje.bar(nombres, valores, color=[GRIS, AZUL], width=0.55)
    for barra, valor in zip(barras, valores):
        eje.text(barra.get_x() + barra.get_width() / 2, valor + 120, f"{valor:,}", ha="center", fontsize=15, weight="bold")
    eje.set_title("Veces en una semana que una estación muy usada se quedó\ncon 2 bicicletas o más de demanda sin anticipar",
                  fontsize=12.5, loc="left")
    eje.set_ylim(0, max(valores) * 1.2)
    eje.set_yticks([])
    eje.spines[["top", "right", "left"]].set_visible(False)
    lineas(figura, [
        f"Qué pasó: en una semana de prueba con 100 estaciones (25 al 31 de agosto), el pronóstico dejó "
        f"{menos:,} veces menos una estación con mucha demanda sin aviso: unas {menos // 7} por día.",
        "Qué recomendamos: un piloto de 4 semanas en estas 100 estaciones, en el que un despachador "
        "ordena las rutas del camión con el pronóstico cada 15 minutos, y medir estaciones vacías y viajes perdidos.",
        "Qué necesitamos: un despachador para el piloto y el registro de los movimientos del camión, "
        "para medir el resultado en estaciones vacías y no solo en el pronóstico.",
        "Qué no hace: no decide la ruta ni cuántas bicicletas llevar; todavía falla en 4 de cada 10 "
        "momentos de mucha demanda; solo está probado con un mes (agosto) y se revisa si cambia la demanda.",
    ])
    pie(figura, f"Momentos de mucha demanda: estación y cuarto de hora con 3 salidas o más en la media hora siguiente "
                f"({cortos['franjas_alta']:,} en la semana). Detalle técnico en docs/resultados_tecnico.pdf.")
    figura.savefig(AQUI / "resultados_ejecutivo.pdf")
    plt.close(figura)


if __name__ == "__main__":
    corridas = pd.read_csv(PROYECTO / "modelado" / "corridas_proyecto.csv").set_index("tags.mlflow.runName")
    corridas = pd.DataFrame({"mae_prueba": corridas.loc[["persistencia 30 min", "ridge", "gradient boosting",
                                                          "red neuronal"], "metrics.mae_prueba"].to_numpy()})
    negocio = pd.read_csv(PROYECTO / "modelado" / "metricas_negocio_proyecto.csv")
    monitoreo_dia = {"mae_semana": 1.0254}  # docs/monitoreo.md, todas las predicciones unidas
    cortos = momentos_cortos()
    print(cortos)
    tecnica(corridas, negocio, monitoreo_dia)
    ejecutiva(cortos)
    print("Guardadas: docs/resultados_tecnico.pdf y docs/resultados_ejecutivo.pdf")
