# Backtesting temporal y calibración probabilística

## Objetivo

Medir y mejorar de forma reproducible la calidad predictiva del simulador de la
Champions 2026/27. El sistema debe seleccionar una calibración de probabilidades
de partido únicamente cuando mejore métricas fuera de muestra; no se promoverá
por apariencia ni por resultados de la temporada en curso.

## Alcance

Esta entrega incluye dos componentes encadenados:

1. Un backtest temporal que recrea cortes históricos antes de cada ventana de
   partidos de Champions y evalúa las predicciones contra los resultados que
   ocurrieron después.
2. Un calibrador multiclase regularizado para probabilidades 1X2, que se ajusta
   solo con bloques temporales de selección y se acepta solo si reduce el
   log-loss ponderado frente al modelo sin calibrar.

No incorpora nuevos proveedores de xG, cuotas futuras, predictores de mercado
ni cambios en la interfaz del dashboard en esta entrega.

## Backtest temporal

El backtest recibirá resultados históricos completos y una competición objetivo.
Agrupará los partidos de Champions por fecha/ventana de jornada. Para cada corte
elegible, entrenará usando exclusivamente encuentros anteriores al corte y
predecirá la siguiente ventana. Nunca se usarán resultados, ratings o variables
con fecha posterior al corte para entrenar o seleccionar modelos.

Cada ventana guardará: fecha de corte, número de partidos de entrenamiento y de
evaluación, probabilidades 1X2, log-loss, Brier multiclase, log-loss de marcador
y diagnóstico de calibración. El informe agregado incluirá medias ponderadas por
partidos y una ventana final de evaluación intacta. Un historial insuficiente
devolverá un estado explícito en lugar de producir una recomendación.

El informe también comparará el modelo de producción sin calibrar y el candidato
calibrado en los mismos cortes. Esta evaluación de partidos es el criterio de
selección; las probabilidades de torneo derivadas de Monte Carlo no se usan para
ajustar el calibrador porque no existen suficientes temporadas equivalentes.

## Calibración y promoción

Se añadirá una transformación multiclase regularizada sobre los logaritmos de
las probabilidades base de local/empate/visitante. La transformación devuelve
tres probabilidades positivas que suman uno y se serializa junto al modelo.
Mantiene el orden relativo cuando la evidencia no justifica corregirlo y evita
la sobreconfianza típica de calibradores no paramétricos con pocas muestras.

La selección temporal separará sus cortes en entrenamiento de calibración,
selección y evaluación final. El calibrador se ajusta en los primeros; se
promueve solo si supera al modelo base en log-loss ponderado de selección por un
umbral mínimo configurable y no empeora Brier. El bloque final se limita a
informar calidad, no a elegir. Si faltan observaciones o clases, se conserva el
modelo base y se registra el motivo.

La calibración se aplicará a la distribución de marcadores de forma compatible:
primero se calcula la probabilidad 1X2 del grid de marcadores; después se reescala
cada celda por el cociente calibrado/base de su resultado y se normaliza. Así la
simulación de fase liga y eliminatorias conserva una distribución de resultados
válida y reproducible.

## Integración y reproducibilidad

`evaluate_models` devolverá una decisión de calibración junto con la actual
decisión entre Poisson, Dixon--Coles y ensemble. `fit_production_model` envolverá
el modelo elegido solo cuando la decisión indique promoción. La carga desde
`model.json` restaurará el envoltorio, por lo que `reproduce` seguirá generando
el mismo snapshot con la misma semilla.

Los metadatos del snapshot guardarán el informe resumido: estado de backtest,
modelo base, decisión, muestras de selección/final y métricas sin calibrar y
calibradas. No se modifica el dashboard minimalista; los detalles se consultan
en el snapshot y en la CLI.

## Errores y límites

- Las validaciones rechazarán probabilidades no finitas, datasets sin resultados
  suficientes y cortes temporales no ordenados.
- La promoción exige mejora fuera de muestra; ante empate, falta de datos o
  regresión se usará el modelo sin calibrar.
- El informe declarará que la validación histórica no convierte el sistema en
  apto para apostar ni garantiza precisión futura.
- La posterior auditoría de ratings (Liverpool, Madrid, AEK y Aston Villa) se
  hará apoyada en el informe, no mediante cambios de fuerza ad hoc.

## Pruebas de aceptación

1. Un fixture sintético demuestra que cada corte solo entrena con partidos
   anteriores y que los resultados futuros no cambian sus predicciones.
2. El calibrador conserva probabilidades válidas, mejora un caso sintético
   descalibrado y deja intacto el modelo cuando no supera el umbral.
3. Un modelo calibrado serializado se restaura y reproduce las mismas
   probabilidades con la misma semilla.
4. La evaluación final no afecta la decisión de promoción.
5. La suite existente continúa pasando.
