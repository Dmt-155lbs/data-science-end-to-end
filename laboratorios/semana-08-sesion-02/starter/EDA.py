import pandas as pd
import os
import matplotlib.pyplot as plt

os.makedirs('./starter/figuras', exist_ok=True)

data = pd.read_csv('./starter/viajes_taxi.csv')

print('Analisis exploratorio descriptivo')

data.info()

print(data.dtypes)
print(data.isna().sum())
print(data.duplicated().sum())
print(data.describe())

plt.figure()
plt.boxplot(data['tarifa'])
plt.title('Boxplot de la tarifa')
plt.savefig('./starter/figuras/boxplot_tarifa.png')
plt.close()

#Extraccion de valores atipicos

q1 = data['tarifa'].quantile(0.25)
q3 = data['tarifa'].quantile(0.75)

iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

print(f'Limite inferior: {limite_inferior}')
print(f'Limite superior: {limite_superior}')

# Filtrar los valores atípicos
atipicos = data.query(f'tarifa < {limite_inferior} or tarifa > {limite_superior}')
sin_atipicos = data.query(f'tarifa >= {limite_inferior} and tarifa <= {limite_superior}')

print(f'Valores atípicos encontrados: {len(atipicos)}')
print(f'Valores sin atípicos: {len(sin_atipicos)}')

plt.figure()
plt.scatter(data['tarifa'], data['distancia_km'])
plt.title('Dispersion de tarifa vs distancia')
plt.xlabel('Tarifa')
plt.ylabel('Distancia (km)')
plt.savefig('./starter/figuras/dispersion_distancia_vs_tarifa.png')
plt.close()

# Correlacion (solo columnas numericas)
print(data.corr(numeric_only=True))

plt.figure()
plt.scatter(sin_atipicos['tarifa'], sin_atipicos['distancia_km'])
plt.title('Dispersion de tarifa vs distancia (sin atípicos)')
plt.xlabel('Tarifa')
plt.ylabel('Distancia (km)')
plt.savefig('./starter/figuras/dispersion_distancia_vs_tarifa_sin_atipicos.png')
plt.close()

# Correlacion (solo columnas numericas)
print(sin_atipicos.corr(numeric_only=True))