# Champions Lab

Simulador de la **UEFA Champions League masculina 2026/27**, en Python y pandas, con dashboard Streamlit y previsiones guardadas por jornada. Los resultados conocidos quedan fijados y se simula el resto de la competición.

## Uso en Windows

Requiere Python 3.11 o posterior. Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m champions.cli update
python -m champions.cli simulate --simulations 20000 --seed 42
python -m streamlit run app/streamlit_app.py
```

Con Python y las dependencias instaladas también puedes abrir `Abrir-dashboard.cmd`. La carpeta `ML CHAMPIONS` del Escritorio enlaza a esta misma copia del proyecto para evitar tener dos versiones divergentes.

Para lanzar una simulación nueva con el número de repeticiones que quieras, usando los datos ya descargados:

```powershell
.\Nueva-simulacion.ps1 -Simulaciones 200000
```

Puedes cambiar también la semilla con `-Semilla 123`. El motor trabaja por lotes, por lo que aumentar las simulaciones incrementa sobre todo el tiempo de cálculo, no la memoria de forma proporcional.

## Después de cada jornada

```powershell
.\Actualizar-jornada.ps1
```

O ejecuta `update` y `simulate` por separado. La actualización es bajo demanda y consulta de nuevo las fuentes; abrir el dashboard no hace descargas. Cada simulación crea un directorio nuevo. No se sobrescriben las probabilidades de semanas anteriores.

`Actualizar-jornada.ps1` descarga los datos nuevos y después simula. `Nueva-simulacion.ps1` solo vuelve a simular los datos locales actuales. Para una reconstrucción histórica, usa un corte UTC explícito:

```powershell
python -m champions.cli simulate --cutoff 2026-09-08T00:00:00Z --matchday 0 --simulations 20000
python -m champions.cli simulate --cutoff 2026-09-16T00:00:00Z --matchday 1 --simulations 20000
python -m champions.cli list
python -m champions.cli reproduce ID_DE_EJECUCION
```

Una reconstrucción usa la versión de datos disponible al descargar: no equivale a una predicción publicada en aquella fecha. Se conservan la fecha del corte y la de creación. Las correcciones retrospectivas de las fuentes pueden cambiar los datos respecto a lo que se conocía entonces.

## Qué muestra

- Top 8: acceso directo a octavos.
- Puestos 9–16 y 17–24: dos grupos del playoff. `playoff` es la probabilidad de acabar 9–24, **no** la de disputar octavos.
- Eliminación en fase liga: puestos 25–36.
- Alcanzar octavos, cuartos, semifinales, final y ser campeón.
- Puntos y posición esperados, clasificación proyectada completa del 1 al 36 y evolución de la posición por corte.

Las columnas de rondas son acumulativas; no deben sumar 100% entre sí. Las categorías top 8 / 9–16 / 17–24 / eliminado sí forman una partición. El error Monte Carlo solo mide precisión numérica de la simulación; no mide todos los errores del modelo.

## Modelos y datos

El modelo de goles aprende fuerza ofensiva y defensiva y ventaja local con regularización y recencia. Se contrasta con un baseline mediante validación temporal y métricas probabilísticas. La documentación de fuentes y modelos se encuentra en `docs/`.

Por defecto, `--model auto` elige entre Poisson y baseline con bloques temporales de Champions; reserva un bloque posterior para evaluación. Puedes fijar `--model poisson` o `--model baseline` para compararlos. No se supone que una métrica obtenida en pocos partidos garantice resultados futuros.

Tras el sorteo de eliminatorias, el motor permite incorporar el cuadro y resultados oficiales con `--knockout RUTA_JSON`; consulta [el esquema](docs/knockout-state.md). La fuente automática actual descarga la fase liga: el cuadro posterior se importa explícitamente cuando exista.

El proyecto conserva los datos descargados localmente. GitHub contiene código y documentación; no redistribuye automáticamente datasets de terceros. Para obtenerlos en otra máquina, ejecuta `update`. Las fuentes, sus hashes, limitaciones y cobertura por club acompañan a los resultados. No hay claves API incrustadas.

## Estructura y reproducibilidad

```text
src/champions/data.py         descarga, normalización y validación
src/champions/models.py       modelos y evaluación temporal
src/champions/simulation.py   reglas de competición y Monte Carlo
src/champions/pipeline.py     snapshots, metadatos y reproducción
src/champions/cli.py          comandos
app/streamlit_app.py          dashboard
data/raw/                    respuestas originales fechadas
data/processed/              última descarga normalizada
data/snapshots/<id>/          datos exactos usados por ejecución
results/snapshots/<id>/       probabilidades, modelo y metadatos
notebooks/explore.py          análisis por celdas
tests/                       pruebas de datos, modelo, motor e interfaz
```

Cada ejecución guarda semilla, parámetros, versión, entorno, corte, evaluación, advertencias, datos utilizados y hashes SHA-256. `reproduce` verifica los hashes y vuelve a simular con el modelo guardado. Para resultados idénticos entre máquinas, conserva las versiones de Python, NumPy, pandas y SciPy indicadas en `metadata.json`.

```powershell
python -m pytest -q
```

El adaptador de datos, el modelo y el motor de competición son independientes, para incorporar otras competiciones sin reescribir el dashboard. Los futuros picks necesitarán cuotas y evaluación de valor esperado; no están implementados en esta versión.
