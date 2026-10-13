# NCAA to NBA Role Predictor

Proyecto de machine learning para estimar el rol NBA temprano de jugadores NCAA a partir de estadisticas universitarias, fisico, contexto de equipo y senales oficiales de participacion cuando estan disponibles.

El proyecto queda cerrado con una salida final de tres clases:

```text
NON_NBA < NBA_ROLE_PLAYER < STARTER_PLUS
```

`NBA_ROLE_PLAYER` agrupa `BENCH` y `ROTATION`. Esa frontera fina resulto demasiado ruidosa para defenderla como decision principal; la version de 3 clases conserva la pregunta util: sin nivel NBA, jugador NBA de rol o perfil de titular/estrella temprana.

## Estado final

- Modelo final: `grouped_xgboost_model`.
- Artefacto local esperado: `models/best_model.pkl`.
- Entrenamiento final: XGBoost multiclase conservador, sin SMOTE, entrenado directamente sobre `NON_NBA`, `NBA_ROLE_PLAYER` y `STARTER_PLUS`.
- Split temporal: train `<= 2018`, validacion `2019-2020`, test final `2021-2022`.
- La cohorte 2023 queda fuera del test final porque su horizonte NBA todavia es menos estable.

Metricas principales del experimento de 3 clases en test:

| Modelo | Accuracy | Balanced Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| XGBoost conservador en 3 clases | 0.9816 | 0.5605 | 0.5992 |
| XGBoost 3 clases con SMOTE | 0.9513 | 0.6775 | 0.4913 |
| Mejor modelo de 4 clases mapeado a 3 | 0.9805 | 0.5571 | 0.5933 |

La version conservadora sin SMOTE es la seleccionada final porque mejora el Macro F1 global y evita falsos positivos en `STARTER_PLUS` en el test temporal. La variante con SMOTE recuperaba mas `STARTER_PLUS`, pero generaba demasiados falsos positivos para una etiqueta que debe leerse como premium.

## Datos y target

El pipeline construye una tabla maestra con jugadores NCAA, informacion de draft, resultados NBA tempranos, medidas fisicas y contexto de equipo.

Fuentes principales:

- College Basketball 2009-2021 + NBA Advanced Stats.
- Basketball Reference Draft.
- 20 Years of NBA Draft Data.
- NBA Draft Combine.
- NBA Dataset Box Scores and Stats.
- NCAA BigQuery Export, cuando exista localmente.
- Bart Torvik y College Basketball Dataset para contexto de equipo.

La tabla maestra actual relabela no drafteados con evidencia NBA verificada por nombre y universidad. Esos datos NBA se usan solo para construir el target, no como features.

La taxonomia fina se conserva internamente para auditoria:

| Clase fina | Clase final |
| --- | --- |
| `NON_NBA` | `NON_NBA` |
| `BENCH` | `NBA_ROLE_PLAYER` |
| `ROTATION` | `NBA_ROLE_PLAYER` |
| `STARTER_PLUS` | `STARTER_PLUS` |

## Que incluye el repositorio

```text
src/
  data/          carga, auditoria y construccion de tabla maestra
  features/      target y features tabulares
  models/        entrenamiento, experimentos y reporte final
  vectorstore/   Milvus, esquemas e ingesta de comparables
  api/           FastAPI
app/             demo Streamlit
docker/          imagenes de datos, ML, API, app y Milvus
tests/           smoke tests y checks de helpers
reports/         solo reportes pequenos seleccionados para trazabilidad
```

No se versionan datasets, credenciales, modelos `.pkl`, datos procesados ni volumenes de Docker.

## Ejecucion

Opcionalmente, copia `.env.example` a `.env` y ajusta las credenciales locales de MinIO antes de levantar Milvus. El repositorio ignora `.env` y `secrets/`.

Validar import de configuracion:

```powershell
docker compose run --rm checkpoint
```

Construir datos procesados:

```powershell
docker compose run --rm data-build-master
```

Entrenar baselines y modelos finos de referencia:

```powershell
docker compose run --rm model-train
```

Entrenar y promocionar el modelo final de 3 clases:

```powershell
docker compose run --rm model-train python -m src.models.bench_rotation_experiment
```

Generar reporte final sincronizado con `models/best_model.pkl`:

```powershell
docker compose run --rm final-report
```

Levantar Milvus, API y demo:

```powershell
docker compose up -d milvus
docker compose run --rm vectorstore-ingest-full
docker compose up -d api
docker compose up -d app
```

URLs locales:

```text
API:  http://localhost:8000/docs
Demo: http://localhost:8501
```

## Limitaciones

- El dataset esta muy desbalanceado y `NON_NBA` domina la cohorte.
- `STARTER_PLUS` tiene pocos ejemplos.
- El modelo cubre rutas NCAA; internacionales y otras rutas quedan fuera.
- No usa video, tracking, mecanica de tiro, defensa sin balon ni contexto tactico avanzado.
- Las predicciones deben interpretarse con probabilidades, comparables historicos y explicabilidad, no como una sentencia unica.
