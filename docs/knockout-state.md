# Actualizaciones durante eliminatorias

Antes del sorteo se simulan todos los cuadros compatibles con los pares de posiciones UEFA. Después del sorteo hay que condicionar por las posiciones y resultados oficiales, sin volver a sortear partidos que ya se conocen.

```powershell
champions simulate --knockout data/processed/knockout.json --simulations 20000
```

Este JSON es una entrada explícita, no un archivo que se inventa a partir de resultados de fase liga. La ingesta automática actual de UEFA cubre la fase liga. El cuadro debe importarse de información oficial cuando exista; aún no ha tenido lugar el sorteo al inicio de la temporada.

## Esquema

- `source`: URL de la información oficial utilizada.
- `as_of`: instante UTC de actualización del cuadro, no posterior al corte de la previsión.
- `league_order`: los 36 nombres canónicos en orden oficial final. El motor comprueba que no contradice puntos, goles y demás criterios disponibles.
- `playoff_seeded`: ocho clubes, posiciones de cada lado [15/16, 9/10, 13/14, 11/12]; primero las cuatro del lado 0 y después las del lado 1.
- `playoff_unseeded`: ocho clubes emparejados con los anteriores, [17/18, 23/24, 19/20, 21/22] por lado.
- `round16_seeds`: opcional antes del sorteo de octavos; ocho clubes [1/2, 7/8, 3/4, 5/6] por lado. Su ausencia genera solo ese sorteo pendiente.
- `ties`: objeto con claves `po-0` a `po-7`, `r16-0` a `r16-7`, `qf-0` a `qf-3`, `sf-0`, `sf-1`, `final-0`.

Cada eliminatoria contiene `legs`, una lista cronológica de los partidos completados; cada partido tiene `home_goals`, `away_goals` de 90 minutos y `completed_at` UTC. Se omiten los partidos pendientes. El campo opcional `winner` indica el ganador oficial y es obligatorio cuando la eliminatoria completada empata tras los 90 minutos de cada partido. Así se respeta el ganador real tras prórroga/penaltis sin usar goles extra como goles reglamentarios.

Se presupone la localía reglamentaria: cabeza de serie recibe la vuelta en playoff/octavos; cuartos y semifinal heredan las prioridades de su posición en el cuadro. Si UEFA modifica una sede u orden excepcionalmente, el formato actual requiere adaptar esa condición antes de usarlo.

El sistema rechaza resultados de una ronda sin sus eliminatorias precursoras completadas, ganadores incompatibles con los participantes, marcadores contradictorios y cuadros posteriores al corte. Un torneo completado devuelve un único campeón con probabilidad uno. El estado utilizado se copia al snapshot, se incluye en los hashes y se vuelve a aplicar al reproducirlo.
