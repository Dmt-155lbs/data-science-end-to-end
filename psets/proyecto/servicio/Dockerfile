# La imagen del servicio de prediccion. Se construye desde psets/proyecto/,
# porque necesita modelado/ ademas de servicio/, y despues de exportar el
# modelo con python servicio/exportar_modelo.py:
#
#     docker build -f servicio/Dockerfile -t servicio-ecobici .
#     docker run -p 8000:8000 servicio-ecobici
FROM python:3.11-slim
WORKDIR /app
COPY servicio/requirements.txt servicio/
RUN pip install --no-cache-dir -r servicio/requirements.txt
COPY modelado/pipeline_proyecto.py modelado/
COPY servicio/api.py servicio/deriva.py servicio/referencia_deriva.json servicio/modelo.joblib servicio/modelo_version.txt servicio/
EXPOSE 8000
CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]
