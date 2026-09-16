# Fuentes y calidad de datos

La descarga `champions update` obtiene respuestas HTTP reales y conserva cada una con fecha, URL, tamaño y SHA-256. El cliente usa certificados del sistema con validación TLS; cada petición tiene su propio contexto para evitar compartir estado SSL entre hilos en Windows.

## Calendario actual

[UEFA: calendario y resultados 2026/27](https://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000--2026-27-champions-league-all-the-league-phase-fixtures/).

Se comprueban los 36 clubes, los 144 partidos, ocho rivales diferentes por club, cuatro localías y cuatro visitas, y una aparición por jornada. Los resultados ausentes permanecen vacíos. No se rellenan con predicciones. Fechas interpretadas según el CET que declara el artículo; los cortes históricos usan un margen conservador de tres horas para evitar tratar como conocido un marcador al comienzo de un partido.

La fuente lista la fase liga. Para actualizar durante eliminatorias se incorpora además el cuadro oficial mediante `--knockout` (ver `knockout-state.md`). La fuente de fase liga no puede determinar los sorteos ni resultados futuros de eliminatorias.

## Histórico de entrenamiento

- [Football-Data.co.uk](https://www.football-data.co.uk/data.php): resultados de Inglaterra, Alemania, Italia, España, Francia, Países Bajos, Bélgica, Portugal, Turquía y Grecia, temporadas 2024/25–2026/27; Austria y Noruega desde sus archivos ampliados.
- [OpenFootball Champions League](https://github.com/openfootball/champions-league): Champions 2024/25 y 2025/26, previas de Champions/Europa/Conference de ambas temporadas y fases principales Europa/Conference 2024/25 disponibles en el repositorio.

Los cruces europeos conectan los equipos de las ligas nacionales. El validador exige historial para todos los participantes y que estén conectados por rivales comunes; no asigna escalas nacionales independientes. Esto no elimina la incertidumbre cuando existen pocos cruces entre ligas.

No están disponibles en esa fuente las fases principales Europa/Conference 2025/26. Tampoco se dispone de la misma profundidad doméstica para todos los países. Sabah y Shakhtar, por ejemplo, tienen una muestra menor y dependen más de la regularización. La cobertura completa del calendario no equivale a una cobertura exhaustiva de la trayectoria de cada club.

Los goles son de 90 minutos. Un resultado con prórroga se incorpora solo si consta explícitamente el marcador al final de los 90 minutos. Los datos sin hora se sitúan conservadoramente al final del día UTC. Las fuentes no suministran uniformemente lesiones, alineaciones, xG ni todas las tarjetas de jugadores y oficiales, por lo que no se inventan esas variables.

## Snapshots y fallos

Cada descarga se publica como una generación completa mediante un puntero atómico. Si falla antes de completarse, sigue activa la anterior. Los resultados históricos se conservan en directorios independientes y son comprobables con `champions reproduce ID`.

Un calendario incompleto detiene la actualización. Los fallos parciales de fuentes históricas aparecen en el informe de calidad y en las advertencias; la falta de un club o de conectividad europea detiene la descarga. Se registran también los equipos con menos de veinte partidos o sin resultados recientes. Revisar estas advertencias antes de interpretar las probabilidades.

Los CSV originales y snapshots permanecen locales y están excluidos de Git. El proyecto enlaza a las fuentes; cualquier redistribución debe respetar las condiciones de UEFA, Football-Data y la licencia del repositorio OpenFootball.
