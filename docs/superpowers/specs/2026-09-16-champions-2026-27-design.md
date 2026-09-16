# Diseño funcional — Simulador UEFA Champions League 2026/27

## 1. Objetivo

Crear una aplicación reproducible en Python para estimar, mediante modelos probabilísticos, las probabilidades de clasificación de cada club en la UEFA Champions League masculina 2026/27.

El sistema se actualizará después de cada jornada con datos recientes, ejecutará nuevas simulaciones y conservará los resultados históricos para comparar cómo evolucionan las previsiones durante la competición.

La arquitectura debe permitir incorporar posteriormente otras competiciones, nuevos modelos y mercados probabilísticos relacionados con apuestas, sin mezclar esas extensiones con el núcleo de la Champions.

## 2. Alcance de la primera versión

### Incluido

- Dataset actualizado de la temporada 2026/27 y datos históricos necesarios para entrenar o calibrar el modelo.
- Actualización bajo demanda después de cada jornada.
- Snapshots fechados de datos, configuración, modelo y resultados.
- Simulación Monte Carlo de la fase liga y eliminatorias.
- Dashboard web sencillo construido con Streamlit.
- Probabilidades por equipo de:
  - terminar en el top 8;
  - terminar entre los puestos 9–16;
  - terminar entre los puestos 17–24 y acceder al playoff (9–24);
  - quedar eliminado en la fase liga;
  - alcanzar octavos de final;
  - alcanzar cuartos de final;
  - alcanzar semifinales;
  - alcanzar la final;
  - ganar la competición.
- Tablas, filtros, ordenación y evolución histórica de las probabilidades.
- Validaciones automáticas de datos y pruebas del pipeline principal.

### Fuera de alcance inicial

- Recomendaciones automáticas de apuestas.
- Conexión directa con casas de apuestas.
- Predicción de cuotas o cálculo de valor esperado.
- Modelos de lesiones, alineaciones o noticias en tiempo real.
- Soporte inicial para otras competiciones.

## 3. Enfoques considerados

### Streamlit + Python — elegido

Permite reutilizar directamente el motor de datos y simulación, crear una interfaz funcional rápidamente y añadir filtros y gráficos sin introducir un frontend independiente.

### Jupyter + Voilà

Adecuado para exploración, pero menos cómodo como aplicación mantenible y menos preparado para crecer.

### FastAPI + frontend independiente

Más flexible a largo plazo, pero añade complejidad innecesaria para la primera versión.

## 4. Arquitectura propuesta

```text
data/
  raw/                 Datos descargados sin transformar
  processed/           Datos normalizados
  snapshots/           Copias fechadas por jornada
src/
  data/                Fuentes, descarga, validación y transformación
  models/              Features, modelos y simulación Monte Carlo
  pipeline/            Actualización, ejecución y persistencia
  config.py            Configuración común
app/
  streamlit_app.py     Punto de entrada del dashboard
results/
  snapshots/           Resultados históricos por ejecución
notebooks/             Análisis exploratorio y validación
tests/                 Pruebas automatizadas
docs/                  Documentación funcional y técnica
```

Las capas se comunicarán mediante estructuras normalizadas y archivos versionables. El dashboard no accederá directamente a fuentes externas: leerá resultados producidos por el pipeline.

## 5. Flujo de datos

1. El usuario ejecuta una actualización indicando temporada y jornada.
2. El sistema descarga o lee el dataset más reciente.
3. Se validan equipos, partidos, fechas, duplicados y valores faltantes.
4. Se normalizan resultados y variables de modelado.
5. Se calculan ratings y variables de forma disponibles hasta la jornada indicada.
6. Se ajusta o calibra el modelo sin utilizar información futura.
7. Se simula el calendario restante miles de veces.
8. Se agregan las posiciones y rondas alcanzadas por equipo.
9. Se guarda un snapshot con datos, parámetros, semilla y resultados.
10. Streamlit carga los snapshots y presenta la información.

## 6. Modelo inicial

La primera implementación debe priorizar interpretabilidad y reproducibilidad. Se propone un modelo probabilístico de goles y resultados, calibrado con datos históricos y variables de fuerza.

Variables candidatas:

- rating de fuerza del equipo;
- forma reciente;
- goles marcados y encajados;
- rendimiento local y visitante;
- ventaja de campo;
- fuerza del rival;
- experiencia histórica, si mejora la calibración.

El partido se simulará a partir de una distribución de goles —por ejemplo Poisson ajustada o una variante bivariada— y se resolverán empates según las reglas de la competición. La fase liga y las eliminatorias se modelarán de acuerdo con el formato oficial de la temporada.

La solución debe permitir sustituir o comparar modelos, por ejemplo:

- baseline de ratings;
- modelo probabilístico de goles;
- modelo ML calibrado;
- ensemble de los anteriores.

Las probabilidades mostradas deberán indicar la versión del modelo y la fecha de actualización.

## 7. Persistencia y reproducibilidad

Cada ejecución guardará como mínimo:

- temporada;
- jornada;
- timestamp de actualización;
- fuente y versión del dataset;
- equipos y partidos usados;
- versión del modelo;
- hiperparámetros;
- número de simulaciones;
- semilla aleatoria;
- resultados agregados;
- advertencias de calidad de datos.

Los resultados no se sobrescribirán. Una ejecución posterior generará un nuevo snapshot asociado a su jornada y fecha.

## 8. Dashboard

La pantalla principal mostrará una tabla con una fila por equipo y columnas de probabilidad para cada hito. Incluirá:

- selector de temporada y jornada;
- ordenación por cualquier probabilidad;
- filtro por equipo;
- gráfico de evolución temporal;
- vista de detalle de un equipo;
- metadatos de la ejecución seleccionada.

El dashboard deberá advertir que las probabilidades son estimaciones estadísticas y no constituyen recomendaciones financieras o de apuestas.

## 9. Calidad, errores y límites

- El pipeline fallará explícitamente si faltan equipos o partidos esenciales.
- Los datos futuros respecto a la jornada seleccionada no podrán entrar en el entrenamiento.
- Se registrarán warnings para valores imputados, fuentes incompletas o cambios de formato.
- Las probabilidades de cada evento deberán estar entre 0 y 1.
- Las relaciones de inclusión —por ejemplo, ganar implica alcanzar la final— se comprobarán automáticamente.
- Las simulaciones deberán ser reproducibles con la misma semilla y configuración.

## 10. Criterios de aceptación

La primera versión se considerará funcional cuando:

1. pueda cargar un snapshot reciente de la Champions 2026/27;
2. valide y normalice los datos sin intervención manual dentro del flujo normal;
3. ejecute una simulación reproducible del calendario restante;
4. genere probabilidades para todos los equipos y rondas definidas;
5. guarde resultados separados por jornada;
6. muestre los resultados en Streamlit;
7. permita comparar al menos dos snapshots;
8. incluya pruebas para validación de datos, simulación y agregación;
9. documente cómo actualizar datos y arrancar el dashboard.

## 11. Evolución prevista

La extensión a otras competiciones se realizará mediante una interfaz común de competición, calendario, reglas de clasificación y modelo. Las futuras funciones de picks deberán construirse sobre probabilidades calibradas y datos de cuotas, manteniendo separadas la predicción deportiva y cualquier cálculo de valor esperado.

## 12. Decisiones concretas de implementación

- Paquete Python `src/champions`: datos, modelos, evaluación, motor, pipeline y CLI separados. Streamlit lee únicamente snapshots locales.
- Datos de fase liga descargados de UEFA y entrenamiento a partir de Football-Data/OpenFootball; cobertura y limitaciones documentadas en `docs/data-sources.md`.
- Modelo de aprendizaje estadístico: Poisson de ataque/defensa, regularizado y ponderado por recencia; comparación con baseline y selección temporal específica de Champions. Detalles en `docs/models.md`.
- La simulación aplica los desempates disponibles, cuadro UEFA, prórroga y penaltis. Los empates que requieren disciplina/coeficientes no disponibles se resuelven aleatoriamente y se contabilizan expresamente; no se presentan como resolución reglamentaria exacta.
- Tras un sorteo, `--knockout` admite el cuadro oficial y resultados conocidos. El adaptador automático actual obtiene la fase liga; la importación del cuadro de eliminatorias es explícita.
- Las actualizaciones por jornada son bajo demanda; no se ha instalado una tarea programada en el ordenador.
- El dataset de calendario se exige completo; el histórico tiene profundidad desigual por club. La aplicación conserva esas advertencias y no equipara número de simulaciones con fiabilidad predictiva.
