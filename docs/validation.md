# Verificación de la entrega

Auditoría realizada el 16 de septiembre de 2026. Los datos y resultados completos permanecen en la carpeta local; los metadatos de cada ejecución son la evidencia primaria y pueden consultarse desde el dashboard.

| Requisito | Evidencia y comprobación |
|---|---|
| Temporada masculina 2026/27 | Calendario descargado directamente de UEFA, 36 equipos, 144 partidos, 18 resultados y 126 pendientes |
| Histórico real reciente | Descarga sin fuentes fallidas, 9.424 registros previos de ligas y competiciones europeas; último partido fechado 14/09/2026 |
| Cobertura | Los 36 clubes tienen historial y están conectados por sus enfrentamientos; se muestran recuentos y advertencias para las muestras pequeñas |
| Modelos aprendidos | Poisson regularizado con recencia frente a baseline; validación temporal general y tres bloques específicos de Champions |
| Ausencia de información futura | Corte UTC, margen de finalización, fechas de disponibilidad conservadas para los cortes internos de validación y pruebas de perturbación del futuro |
| Reproducibilidad | Semilla, configuración, modelo JSON, entorno, código, hashes de datos y resultados; comando `reproduce` ejecutado sobre ambos cortes |
| Probabilidades y formato UEFA | Pruebas de conservación de plazas, inclusión de rondas, tabla/desempates, cuadro reglamentario, prórroga, final neutral y penaltis |
| Marcadores conocidos | Fijos en las simulaciones; soporte de cuadro oficial y partidos de eliminatorias mediante estado explícito |
| Historial | Dos cortes reales: previo al inicio y después de jornada 1, con 20.000 torneos simulados en cada uno; ejecuciones inmutables |
| Dashboard | Pruebas Streamlit AppTest sin acceso a red y revisión visual en navegador local; tabla, filtros, evolución, comparación, cobertura e incertidumbre |
| Actualización | Descarga validada publicada mediante puntero atómico; script `Actualizar-jornada.ps1` para actualizar y volver a simular |
| Documentación | README y documentos de fuentes, modelos, competición, cuadro de eliminatorias y diseño funcional |
| Escritorio y GitHub | Carpeta `ML CHAMPIONS` en el Escritorio enlazada a la copia de trabajo; repositorio privado `martiinbz/champions-lab` |

## Límites de la evidencia

La cobertura del calendario está validada; no se afirma que el histórico contenga todos los partidos domésticos posibles de cada equipo. La muestra europea de Sabah y Shakhtar es menor que la de clubes de las ligas principales. El modelo no tiene lesiones, alineaciones ni una cobertura uniforme de xG.

Las métricas se calculan sobre resultados fuera del entrenamiento. Los bloques de selección no se presentan como una prueba independiente; se reserva el último bloque. El número de partidos del bloque final y las curvas de calibración están en los metadatos. Estos resultados no garantizan rendimiento futuro.

La actualización automática descarga la fase liga. Después del sorteo, el cuadro oficial se incorpora con `--knockout`; no se deducen cruces conocidos a partir de un cuadro aleatorio. Excepciones de sede u orden de partidos decididas por UEFA requieren reflejarse en el motor antes de usarlas.

Si una simulación empata en todos los criterios de clasificación disponibles y faltan disciplina/coeficientes, se usa un desempate aleatorio contabilizado en los diagnósticos. La clasificación final oficial puede fijarse en el estado de eliminatorias. El error Monte Carlo y la incertidumbre del modelo se explican por separado.
