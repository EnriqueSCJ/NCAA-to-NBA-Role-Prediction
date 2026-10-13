"""FastAPI application for NCAA to NBA role prediction."""

from __future__ import annotations

import pickle
import sys
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pymilvus import MilvusClient

import src.config as config
from src.api.schemas import (
    ComparablePlayer,
    ComparablePlayersResponse,
    HealthResponse,
    PlayerStats,
    PredictionResponse,
)
from src.features.build_features import (
    apply_z_score_stats,
    build_context_features,
    build_ncaa_features,
    build_physical_features,
    combine_all_features,
)
from src.models.train import FEATURE_COLUMNS, XGBoostRoleClassifier
from src.models.guarded_model import StarterPlusFamilyVetoRoleClassifier, StarterPlusGuardedRoleClassifier
from src.models.grouped_model import GroupedXGBoostClassifier, RoleGroupCascadeClassifier
from src.models.probability_adjusted_model import ProbabilityAdjustedRoleClassifier
from src.models.two_stage_model import BinaryXGBoostClassifier, TwoStageRoleClassifier
from src.vectorstore.ingest import get_milvus_uri, make_demo_master_table
from src.vectorstore.ingest import normalize_vectors
from src.vectorstore.search import search_comparable_players


app = FastAPI(
    title="NCAA to NBA Role Predictor",
    description="Predicts early NBA role and retrieves comparable historical profiles.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL: Any | None = None
MODEL_NAME: str | None = None
MODEL_LOAD_ERROR: str | None = None


def _model_path() -> Path:
    """Return the path to the selected model artifact."""
    return config.MODELS_DIR / "best_model.pkl"


def _register_pickle_aliases() -> None:
    """Register model classes that may have been pickled from __main__."""
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        setattr(main_module, "XGBoostRoleClassifier", XGBoostRoleClassifier)
        setattr(main_module, "StarterPlusGuardedRoleClassifier", StarterPlusGuardedRoleClassifier)
        setattr(main_module, "StarterPlusFamilyVetoRoleClassifier", StarterPlusFamilyVetoRoleClassifier)
        setattr(main_module, "GroupedXGBoostClassifier", GroupedXGBoostClassifier)
        setattr(main_module, "RoleGroupCascadeClassifier", RoleGroupCascadeClassifier)
        setattr(main_module, "ProbabilityAdjustedRoleClassifier", ProbabilityAdjustedRoleClassifier)
        setattr(main_module, "BinaryXGBoostClassifier", BinaryXGBoostClassifier)
        setattr(main_module, "TwoStageRoleClassifier", TwoStageRoleClassifier)


def _infer_model_name(model: Any) -> str:
    """Infer a readable model name from the loaded estimator."""
    if isinstance(model, TwoStageRoleClassifier):
        return "two_stage_role_model"
    if isinstance(model, RoleGroupCascadeClassifier):
        return "grouped_to_fine_cascade"
    if isinstance(model, GroupedXGBoostClassifier):
        return "grouped_xgboost_model"
    if isinstance(model, StarterPlusFamilyVetoRoleClassifier):
        return "xgboost_starter_plus_family_veto"
    if isinstance(model, StarterPlusGuardedRoleClassifier):
        if float(getattr(model, "minimum_starter_plus_probability", 0.0)) > 0.0:
            return "xgboost_starter_plus_threshold_guarded"
        return "xgboost_starter_plus_guarded"
    if isinstance(model, XGBoostRoleClassifier):
        return "xgboost_model"
    if isinstance(model, ProbabilityAdjustedRoleClassifier):
        return "xgboost_probability_adjusted"
    return type(model).__name__


def load_model() -> Any:
    """Load the persisted best model, caching it for future requests."""
    global MODEL, MODEL_NAME, MODEL_LOAD_ERROR

    if MODEL is not None:
        return MODEL

    path = _model_path()
    if not path.exists():
        MODEL_LOAD_ERROR = f"Model artifact not found: {path}"
        raise FileNotFoundError(MODEL_LOAD_ERROR)

    try:
        _register_pickle_aliases()
        with path.open("rb") as file:
            MODEL = pickle.load(file)
        MODEL_NAME = _infer_model_name(MODEL)
        MODEL_LOAD_ERROR = None
        return MODEL
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        raise


@app.on_event("startup")
def startup_event() -> None:
    """Load model once when the API starts."""
    try:
        load_model()
    except Exception as exc:
        print(f"Model could not be loaded at startup: {exc}")


def _dump_player_stats(stats: PlayerStats) -> dict[str, Any]:
    """Return a dict for Pydantic v1 and v2."""
    if hasattr(stats, "model_dump"):
        return stats.model_dump()
    return stats.dict()


def _player_id_from_name(name: str, season: int) -> str:
    """Create a stable API player id."""
    slug = "".join(char.lower() if char.isalnum() else "_" for char in name)
    slug = "_".join(part for part in slug.split("_") if part)
    return f"api_{slug}_{season}"[:256]


def player_stats_to_row(stats: PlayerStats) -> dict[str, Any]:
    """Convert API input into the raw table format expected by feature builders."""
    values = _dump_player_stats(stats)
    season = int(values.get("season") or values.get("draft_year") or 2021)
    wingspan = values.get("wingspan_in") or float(values["height_in"]) + 3.0
    ast_tov_ratio = values.get("ast_tov_ratio")
    if ast_tov_ratio is None:
        turnovers = float(values.get("tov_per40") or 0.0)
        ast_tov_ratio = float(values["ast_per40"]) / turnovers if turnovers else 0.0

    return {
        **values,
        "player_id": _player_id_from_name(values["player_name"], season),
        "season": season,
        "wingspan_in": wingspan,
        "ast_tov_ratio": ast_tov_ratio,
        "stocks_per40": float(values.get("stl_per40", 0.0)) + float(values.get("blk_per40", 0.0)),
        "bmi": (float(values["weight_lb"]) / (float(values["height_in"]) ** 2)) * 703.0,
        "wingspan_diff": float(wingspan) - float(values["height_in"]),
        "target_role": "UNKNOWN",
    }


def load_reference_table() -> tuple[pd.DataFrame, str]:
    """Load the real master table if present, otherwise a demo reference cohort."""
    master_path = config.DATA_PROCESSED / "master_table.csv"
    if master_path.exists():
        reference = pd.read_csv(master_path, low_memory=False)
        if "player_id" not in reference.columns:
            reference = reference.copy()
            reference["player_id"] = [f"reference_{index}" for index in range(len(reference))]
        return reference, "processed_master_table"

    return make_demo_master_table(), "demo_reference_table"


def load_feature_scaler_stats() -> dict[str, dict[str, float]]:
    """Load train-fit z-score stats for API feature preparation."""
    path = config.DATA_PROCESSED / "feature_scaler.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    stats = payload.get("stats", {})
    return stats if isinstance(stats, dict) else {}


def build_prediction_features(stats: PlayerStats) -> tuple[pd.DataFrame, str]:
    """Build the one-row feature matrix consumed by the trained model."""
    api_row = player_stats_to_row(stats)
    reference, source = load_reference_table()
    raw = pd.concat([pd.DataFrame([api_row]), reference], ignore_index=True, sort=False)

    ncaa_features = build_ncaa_features(raw)
    physical_features = build_physical_features(raw)
    context_features = build_context_features(raw)
    combined = combine_all_features(ncaa_features, physical_features, context_features)
    scaler_stats = load_feature_scaler_stats()
    if scaler_stats:
        combined = apply_z_score_stats(combined, scaler_stats)

    player_id = str(api_row["player_id"])
    matching_rows = combined[combined["player_id"] == player_id]
    if matching_rows.empty:
        raise ValueError("Could not build feature row for API player")

    features = matching_rows.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    features = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.astype(float), source


def build_comparable_query_vector(stats: PlayerStats) -> tuple[list[float], str]:
    """Build a Milvus numeric-vector query from an arbitrary NCAA player profile."""
    api_row = player_stats_to_row(stats)
    reference, source = load_reference_table()
    raw = pd.concat([pd.DataFrame([api_row]), reference], ignore_index=True, sort=False)
    vectors = normalize_vectors(raw)
    if not vectors:
        raise ValueError("Could not build comparable-search vector")
    return vectors[0], source


def _classes_from_model(model: Any, probabilities: np.ndarray) -> list[str]:
    """Return class labels aligned with a probability vector."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = config.ROLE_LABELS

    labels = [str(label) for label in list(classes)]
    if len(labels) == len(probabilities):
        return labels

    return config.ROLE_LABELS[: len(probabilities)]


def predict_from_features(
    model: Any,
    features: pd.DataFrame,
    player_name: str,
    feature_source: str,
) -> PredictionResponse:
    """Run the model and format a prediction response."""
    raw_probabilities = np.asarray(model.predict_proba(features))
    if raw_probabilities.ndim == 1:
        probabilities = raw_probabilities
    else:
        probabilities = raw_probabilities[0]

    classes = _classes_from_model(model, probabilities)
    probability_map = {
        label: float(probability)
        for label, probability in zip(classes, probabilities)
    }
    predicted_role = str(classes[int(np.argmax(probabilities))])
    confidence = float(np.max(probabilities))

    return PredictionResponse(
        player_name=player_name,
        predicted_role=predicted_role,
        probabilities=probability_map,
        confidence=confidence,
        model_name=MODEL_NAME or _infer_model_name(model),
        feature_source=feature_source,
    )


@app.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return API health and model status."""
    return HealthResponse(
        status="API running" if MODEL_LOAD_ERROR is None else f"API running, model error: {MODEL_LOAD_ERROR}",
        model_loaded=MODEL is not None,
        model_name=MODEL_NAME,
        milvus_collection=config.MILVUS_COLLECTION_NAME,
    )


@app.post("/predict-role", response_model=PredictionResponse)
def predict_role(stats: PlayerStats) -> PredictionResponse:
    """Predict a player's early NBA role."""
    try:
        model = load_model()
        features, feature_source = build_prediction_features(stats)
        return predict_from_features(model, features, stats.player_name, feature_source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


def _escape_milvus_string(value: str) -> str:
    """Escape a string for simple Milvus equality filters."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _comparable_service_unavailable(exc: Exception) -> HTTPException:
    """Return a clear API error when Milvus has not been ingested yet."""
    detail = str(exc)
    if "collection does not exist" in detail.lower():
        detail = (
            f"Milvus collection '{config.MILVUS_COLLECTION_NAME}' does not exist. "
            "Run `docker compose run --rm vectorstore-ingest-full` before using "
            "the comparable-player search."
        )
    return HTTPException(status_code=503, detail=detail)


def _find_player_in_milvus(player_name: str) -> dict[str, Any] | None:
    """Find a player row in Milvus by exact or case-insensitive name."""
    client = MilvusClient(uri=get_milvus_uri())
    collection_name = config.MILVUS_COLLECTION_NAME
    if not client.has_collection(collection_name):
        return None

    client.load_collection(collection_name)
    output_fields = [
        "player_id",
        "player_name",
        "numeric_vector",
        "position",
        "target_role",
        "draft_year",
        "college",
    ]

    try:
        exact_matches = client.query(
            collection_name=collection_name,
            filter=f'player_name == "{_escape_milvus_string(player_name)}"',
            output_fields=output_fields,
            limit=1,
        )
        if exact_matches:
            return exact_matches[0]
    except Exception:
        pass

    candidates = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=output_fields,
        limit=1000,
    )
    normalized_name = player_name.strip().lower()
    for candidate in candidates:
        candidate_name = str(candidate.get("player_name", "")).strip().lower()
        if candidate_name == normalized_name:
            return candidate

    for candidate in candidates:
        candidate_name = str(candidate.get("player_name", "")).strip().lower()
        if normalized_name and normalized_name in candidate_name:
            return candidate

    return None


@app.get("/comparable-players/{player_name}", response_model=ComparablePlayersResponse)
def comparable_players(
    player_name: str,
    top_k: int = Query(5, ge=1, le=20),
) -> ComparablePlayersResponse:
    """Return Milvus comparable players for an indexed player name."""
    matched = _find_player_in_milvus(player_name)
    if matched is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Player not found in Milvus. Run vectorstore-ingest-full and "
                "query a player present in the collection."
            ),
        )

    query_vector = matched.get("numeric_vector")
    if not isinstance(query_vector, list):
        raise HTTPException(status_code=500, detail="Matched player has no numeric vector")

    raw_results = search_comparable_players(query_vector=query_vector, top_k=top_k + 1)
    matched_id = matched.get("player_id")
    comparables = [
        ComparablePlayer(**result)
        for result in raw_results
        if result.get("player_id") != matched_id
    ][:top_k]

    return ComparablePlayersResponse(
        player_name=str(matched.get("player_name", player_name)),
        matched_player_id=str(matched_id) if matched_id is not None else None,
        top_k=top_k,
        comparables=comparables,
    )


@app.post("/comparable-players/profile", response_model=ComparablePlayersResponse)
def comparable_players_for_profile(
    stats: PlayerStats,
    top_k: int = Query(default=5, ge=1, le=20),
) -> ComparablePlayersResponse:
    """Return historical comparables for an arbitrary NCAA profile."""
    try:
        query_vector, _ = build_comparable_query_vector(stats)
        raw_results = search_comparable_players(query_vector=query_vector, top_k=top_k)
    except ValueError as exc:
        raise _comparable_service_unavailable(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparable search failed: {exc}") from exc

    comparables = [ComparablePlayer(**result) for result in raw_results[:top_k]]
    return ComparablePlayersResponse(
        player_name=stats.player_name,
        matched_player_id=None,
        top_k=top_k,
        comparables=comparables,
    )


def main() -> None:
    """Run the API with uvicorn for local execution."""
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
