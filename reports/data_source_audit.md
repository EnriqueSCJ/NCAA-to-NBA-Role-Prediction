# Auditoria de fuentes disponibles en `Datasets`

Fecha: 2026-05-25

## Resumen ejecutivo

Si se puede mejorar la calidad de los datos con lo que ya existe en `Datasets`, pero no conviene meter todas las tablas como features. La mejora mas importante no es anadir mas columnas: es corregir etiquetas de no drafteados que aparecen en datos NBA reales.

El pipeline actual usa bastantes fuentes, pero todavia tiene oportunidades:

- Usa NCAA 2009-2021, Bart Torvik 2022-2023, draft 2009-2023, RAPTOR, box scores NBA, Combine Kaggle, V2 NBA Players, College Basketball Dataset y BigQuery player games.
- No usa de forma productiva `CollegeBasketballPlayers2022.csv`, las pruebas atleticas del Combine, `NBA Dataset ... / Players.csv` como fuente de identidad, `*_team_results.csv` como features historicas, ni PlayByPlay.
- Varias fuentes NBA son peligrosas como features porque contienen informacion posterior al draft. Si se usan, debe ser para construir target, validar identidad o rellenar medidas fisicas, no para predecir.

El reporte reproducible esta en `reports/data_source_audit.json` y se genera con:

```powershell
docker compose run --rm model-train python -m src.data.audit_dataset_sources
```

## Hallazgo principal: no drafteados mal etiquetados

La auditoria inicial encontro que la tabla tenia 26,129 filas no drafteadas y todas se forzaban a `NON_NBA` upstream. Eso no era siempre cierto: algunos jugadores no drafteados acaban jugando en NBA.

La correccion ya esta aplicada en `src/data/build_master.py`: se relabela un no drafteado solo si hay actividad NBA posterior y match conservador de nombre + universidad contra `NBA Dataset Box Scores and Stats/Players.csv`.

La auditoria detecto:

| Medicion | Filas |
| --- | ---: |
| No drafteados en master | 26,129 |
| No drafteados con algun match de nombre en NBA | 627 |
| No drafteados con actividad NBA posterior a NCAA | 567 |
| Cambiarian de `NON_NBA` si se relabela solo por nombre | 205 |
| Cambian con match conservador de universidad | 155 |

Distribucion de los 155 cambios con match conservador de universidad:

| Nuevo rol | Filas |
| --- | ---: |
| `BENCH` | 87 |
| `ROTATION` | 64 |
| `STARTER_PLUS` | 4 |

Tambien habia 28 casos adicionales con nombre no ambiguo en el master, pero sin match de universidad. Esos quedan como candidatos a revision, no como relabel automatico.

Importante: el match solo por nombre no es suficientemente seguro. El auditor encontro homonimos claros, por ejemplo filas universitarias cuyo `college` no coincide con el `school` del jugador NBA. Por eso la correccion productiva usa nombre + universidad compatible, y deja los ambiguos fuera.

Tras aplicar la correccion, el auditor encuentra 517 outcomes de undrafted que ya coinciden con el target actual y 50 candidatos que seguirian cambiando si se aceptaran matches menos estrictos. Esos 50 no se promocionan automaticamente.

## NCAA 2022

`CollegeBasketballPlayers2022.csv` existe pero el pipeline actual usa `BartTorvik2022.csv`.

| Comparacion 2022 | Filas |
| --- | ---: |
| `CollegeBasketballPlayers2022.csv` | 3,978 |
| `BartTorvik2022.csv` | 5,012 |
| Solape jugador-equipo-temporada | 3,402 |
| Solo en `CollegeBasketballPlayers2022.csv` | 576 |
| Solo en `BartTorvik2022.csv` | 1,610 |

Hay 576 filas 2022 que hoy no entran desde el CSV NCAA 2022. Aun asi, meterlas directamente probablemente anadiria sobre todo negativos al test 2022. Tiene sentido revisarlas despues de corregir el target de no drafteados, no antes.

## Combine

El pipeline ya usa del Combine:

- altura (`HGT`)
- peso (`WGT`)
- wingspan (`WNGSPN`)
- posicion (`POS`)
- BMI (`BMI`)

Pero ignora pruebas atleticas pre-draft:

- `STNDVERT`
- `LPVERT`
- `LANE`
- `SHUTTLE`
- `SPRINT`
- `BENCH`

Cobertura exacta por jugador/anio en el master:

| Split | Filas cubiertas |
| --- | ---: |
| Train 2009-2018 | 507 / 16,964 |
| Validacion 2019-2020 | 91 / 3,418 |
| Test 2021-2022 | 97 / 6,407 |
| Exploratorio 2023 | 37 / 45 |

Estas pruebas pueden aportar senial para drafteados, pero la missingness no es aleatoria: ser invitado al Combine ya es una senial de scouting. Si se anaden, deben ir con indicadores de missingness y validacion temporal/ablacion.

La fuente alternativa `NBA_Draft_Combine_2000_2024_nl.csv` no aporta matches exactos adicionales frente al Combine Kaggle en el master actual.

## Contexto de equipo

Los `*_team_results.csv` de Bart Torvik tienen variables de calendario/contexto mas ricas:

- `sos`
- `ncsos`
- `consos`
- `Opp OE`
- `Opp DE`
- `Con Adj OE`
- `Con Adj DE`
- `WAB`
- `adjt`

Pero en la carpeta solo existen para 2022-2026. En 2022 y 2023 cubren el 100% de las filas master de esos anios, pero no cubren entrenamiento historico. Anadir columnas que solo existen en test/futuro crearia un artefacto de cohorte. Son utiles para la app/inferencia reciente, pero no como features supervisadas nuevas salvo que se consiga cobertura equivalente para train.

## BigQuery NCAA

El parquet `mbb_players_games_sr` tiene 888,844 filas, pero las temporadas disponibles tras el shift actual son solo 2014-2018.

Cobertura de features oficiales actuales:

- 2009-2013: 0%
- 2014-2018: 86%-99%
- 2019-2023: 0%

Esto limita mucho cualquier mejora basada en BigQuery con los archivos actuales. Derivar mas estadisticas oficiales de ese parquet podria enriquecer train, pero no validacion/test; por tanto puede introducir covariate shift. Para que BigQuery sea una mejora limpia haria falta exportar temporadas recientes equivalentes o hacer una ablacion seria de estas columnas.

## Fuentes NBA auxiliares

`NBA Dataset Box Scores and Stats (1947 - Today)/Players.csv` tiene 6,692 filas y 1,911 matches por nombre contra el master. Es util para:

- validar identidad de jugadores no drafteados mediante `school`;
- rellenar altura/peso (`heightInches`, `bodyWeightLbs`);
- auditar metadata de draft.

No debe usarse como feature con `draftYear`, `draftRound`, `draftNumber`, `fromYear`, `toYear` ni career stats, porque eso cambia el problema o mete informacion posterior.

`PlayerStatisticsExtended.csv` ya se usa para target. Se podria mejorar el proxy de impacto con mas columnas de esa misma fuente, pero seguiria siendo target-side, no feature-side.

`PlayByPlay.parquet` no debe entrar como feature pre-draft. Podria servir para labels mas sofisticadas, pero es pesado y no parece la primera mejora costo/beneficio.

## Recomendacion

Orden recomendado:

1. Mantener la correccion de target de no drafteados con match conservador nombre + universidad compatible.
2. Conservar `xgboost_model` sin SMOTE como artefacto final mientras sea el mejor por Macro F1 de validacion.
3. Hacer una ablacion mas formal de features oficiales BigQuery si se decide cambiar el contrato de features.
4. Probar Combine atletico con indicadores de missingness y seleccion por validacion, no por test.
5. Revisar las 576 filas unicas de `CollegeBasketballPlayers2022.csv` solo despues de estabilizar labels; si entran antes, probablemente solo inflan negativos del test.

Conclusion: si hay mejores datos dentro de `Datasets`, pero la mejora defendible empieza por labels e identidad, no por anadir todas las columnas disponibles.
