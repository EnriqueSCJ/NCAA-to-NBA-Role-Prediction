# Model Card - NCAA to NBA Role Predictor

## Objetivo

Este modelo estima el rol NBA temprano de un jugador a partir de su perfil NCAA: produccion por 40 minutos, eficiencia, fisico y contexto competitivo del equipo. La salida final del proyecto usa tres clases:

```text
NON_NBA < NBA_ROLE_PLAYER < STARTER_PLUS
```

`NBA_ROLE_PLAYER` agrupa las etiquetas finas `BENCH` y `ROTATION`, que eran la frontera mas ruidosa del problema. La taxonomia fina se conserva para construir el target y documentar experimentos, pero la API y la demo deben leerse en la salida de 3 clases.

El sistema esta pensado como herramienta academica y de exploracion: acompania la prediccion con comparables en Milvus y explicaciones SHAP, no como sustituto de scouting humano.

## Arquitectura

- Modelo seleccionado: `grouped_xgboost_model`.
- Entrenamiento con split temporal para reducir fuga de informacion.
- El modelo final entrena directamente `NON_NBA`, `NBA_ROLE_PLAYER` y `STARTER_PLUS` sin SMOTE, para tratar `STARTER_PLUS` como una etiqueta conservadora.
- La salida fina de 4 clases queda como material comparativo e historico.
- Se probaron variantes con SMOTE, dos etapas, guardarrailes para `STARTER_PLUS`, ajuste de probabilidades, embeddings y modelos ligeros.
- Variables normalizadas con z-score ajustado solo en entrenamiento y reutilizadas por API, entrenamiento y busqueda vectorial.
- Milvus almacena vectores numericos y texto narrativo para recuperar perfiles historicos similares.

## Dataset

| Elemento | Valor |
| --- | ---: |
| Filas totales | 26,834 |
| Filas drafteadas enlazadas | 705 |
| Filas no drafteadas | 26,129 |
| Anio minimo modelado | 2009 |
| Anio maximo modelado | 2023 |
| Variables de entrada | 29 |

Procedencia de draft:

| Fuente | Filas |
| --- | ---: |
| Basketball Reference Draft | 88 |
| Excel historico NCAA/NBA 2009-2021 | 4 |
| No drafteados | 26,129 |
| 20 Years of NBA Draft Data | 613 |

Distribucion global del target:

| Rol | Filas |
| --- | ---: |
| `NON_NBA` | 26142 |
| `BENCH` | 355 |
| `ROTATION` | 293 |
| `STARTER_PLUS` | 44 |

Correccion de no drafteados con evidencia NBA:

| Rol tras relabel conservador | Filas |
| --- | ---: |
| `NON_NBA` | 267 |
| `BENCH` | 87 |
| `ROTATION` | 64 |
| `STARTER_PLUS` | 4 |

El relabel de no drafteados usa datos NBA solo para construir el target, no como variables de entrada. La correccion exige match de nombre y universidad compatible contra `NBA Dataset Box Scores and Stats/Players.csv`, y despues calcula el outcome con las mismas reglas de primeras temporadas activas usadas para drafteados.

Split temporal:

| Bloque | Rango |
| --- | --- |
| Entrenamiento | `draft_year <= 2018` |
| Validacion | `2019-2020` |
| Test final | `2021-2022` |

Distribucion del test en la evaluacion final:

| Rol | Filas |
| --- | ---: |
| `NON_NBA` | 6294 |
| `NBA_ROLE_PLAYER` | 108 |
| `STARTER_PLUS` | 5 |

## Metricas del mejor modelo

| Metrica | Valor |
| --- | ---: |
| Accuracy | 0.9816 |
| Balanced Accuracy | 0.5605 |
| Macro F1 | 0.5992 |
| Weighted F1 | 0.9817 |
| Top-2 Accuracy | 0.9995 |
| Log Loss | 0.0596 |

Comparacion de modelos:

| Modelo | Validacion Macro F1 | Test Accuracy | Test Balanced Accuracy | Test Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| grouped_xgboost_model | 0.6374 | 0.9816 | 0.5605 | 0.5992 |
| xgboost_model | 0.5707 | 0.9774 | 0.4144 | 0.4468 |
| catboost_model | 0.4512 | 0.8740 | 0.5381 | 0.3273 |
| xgboost_smote_model | 0.4092 | 0.9307 | 0.4915 | 0.3400 |
| lr_baseline | 0.3347 | 0.8840 | 0.4889 | 0.2958 |
| rf_baseline | 0.3241 | 0.9062 | 0.5703 | 0.3134 |
| dummy_baseline | 0.2464 | 0.9824 | 0.2500 | 0.2478 |

## Limitaciones

- El problema esta fuertemente desbalanceado: `NON_NBA` domina la cohorte.
- La clase `STARTER_PLUS` agrupa titulares y estrellas para evitar una division artificial con pocos casos.
- Los jugadores internacionales o rutas no NCAA quedan fuera del alcance del predictor tabular.
- Las cohortes recientes deben tratarse con cuidado: 2023 queda fuera del test final porque su horizonte NBA es menos estable.
- Las predicciones deben interpretarse junto con probabilidades, SHAP y comparables, no como una sentencia unica.

## Diagnostico de la matriz de confusion

La matriz de confusion final se lee sobre tres clases. Agrupar `BENCH` y `ROTATION` reduce el ruido de una frontera que no era estable en test y centra la decision en la pregunta defendible: sin nivel NBA, jugador NBA de rol o perfil de titular/estrella temprana.

La auditoria de features muestra que las variables oficiales de BigQuery aportan algo de senial, pero su cobertura se concentra en 2014-2018. Por eso quedan documentadas como senial util con riesgo de covariate shift.

## Mapeo de etiquetas

| Clase fina | Clase agrupada |
| --- | --- |
| `NON_NBA` | `NON_NBA` |
| `BENCH` | `NBA_ROLE_PLAYER` |
| `ROTATION` | `NBA_ROLE_PLAYER` |
| `STARTER_PLUS` | `STARTER_PLUS` |

El modelo final se entrena ya sobre la clase agrupada, no solo como un mapeo posterior.

## Enfoque Comparativo

El proyecto conserva el enfoque en dos etapas como comparacion metodologica. La arquitectura alternativa sigue siendo util para analizar el problema:

| Etapa | Pregunta | Target |
| --- | --- | --- |
| 1 | Tiene perfil NBA? | `NON_NBA` vs `NBA` |
| 2 | Si tiene perfil NBA, que rol alcanza? | `BENCH`, `ROTATION`, `STARTER_PLUS` |

Esta separacion evita que `NON_NBA` inunde la decision de rol. En la ejecucion actual queda como comparativa, no como modelo final.

## Uso Recomendado

El uso mas defendible es mostrar la salida de 3 clases junto con probabilidades, comparables y explicaciones. En presentacion academica, el valor principal esta en la trazabilidad del flujo completo: fuentes, target, features, modelo, explicabilidad y recuperacion vectorial.
