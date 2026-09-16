# Champions Lab Implementation Plan

**Goal:** implementar el diseño aprobado con datos reales, modelos evaluados temporalmente y resultados reproducibles.

**Architecture:** paquete `src/champions` con adaptadores de datos, modelos de goles, motor de competición y pipeline independiente. Streamlit consulta snapshots inmutables. Los módulos se revisan y prueban antes de integrarse.

**Tech Stack:** Python, pandas, NumPy, SciPy, scikit-learn, Streamlit, Plotly, pytest.

- [x] Datos (`data.py`, `tests/test_data.py`): descargar calendario UEFA 2026/27 e históricos; normalizar nombres; verificar 36 clubes, 144 partidos y 4 partidos en casa/4 fuera por club. Ejecutar `python -m pytest tests/test_data.py` y descarga real con informe de cobertura.
- [x] Modelos (`models.py`, `tests/test_models.py`): Poisson ataque/defensa regularizado y ponderado por recencia; baseline; evaluación temporal con log loss/Brier/calibración; persistencia JSON. Probar exclusión de datos futuros y predicciones para equipos escasos.
- [x] Competición (`simulation.py`, `tests/test_simulation.py`): clasificación con desempates UEFA, playoff 9–24 y bracket reglamentario hasta final; resultados conocidos fijos; prórroga/penaltis. Verificar conservación de plazas y reproducibilidad.
- [x] Pipeline (`pipeline.py`, `cli.py`, `tests/test_pipeline.py`): actualización explícita, cutoff temporal, snapshots atómicos e inmutables con hashes, datos/modelo/config/semilla/resultados y métricas.
- [x] Dashboard (`app/streamlit_app.py`, `tests/test_dashboard.py`): tabla ordenable, temporada/jornada/ejecución, equipo, evolución y comparación entre snapshots; metadatos y límites del modelo.
- [x] Integración: ejecutar sobre datos actuales, crear dos cortes temporales reales, arrancar Streamlit, comprobar interfaz y pruebas completas.
- [x] Entrega: README en español, documento funcional actualizado, carpeta del Escritorio y repositorio GitHub privado. Verificar remote y archivos publicados; datos descargados permanecen locales con comandos reproducibles.

## Criterios de auditoría

La existencia de una interfaz o pruebas sintéticas no acredita disponibilidad de datos actuales. Se requiere descargar datos reales y comprobar cobertura por club y fecha, integridad del calendario y calidad predictiva fuera de muestra. Cualquier limitación se registra en los metadatos y la documentación; no se sustituyen datos reales por datos inventados.
