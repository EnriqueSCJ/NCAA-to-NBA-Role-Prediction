"""Create and populate the Milvus player collection."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from pymilvus import MilvusClient

import src.config as config
from src.vectorstore.schema import get_collection_schema, get_index_params


NUMERIC_VECTOR_COLUMNS = list(config.MODEL_FEATURE_COLUMNS)


FIELD_ALIASES = {
    "player_id": ["player_id", "pid", "id"],
    "player_name": ["player_name", "name", "Player", "PLAYER", "player"],
    "season": ["season", "year", "draft_year"],
    "college": ["college", "team", "TEAM", "school"],
    "position": ["position", "pos", "POS"],
    "height_in": ["height_in", "HGT", "height", "ht_in"],
    "weight_lb": ["weight_lb", "WGT", "weight", "wt"],
    "draft_year": ["draft_year", "year", "season"],
    "draft_pick": ["draft_pick", "pick", "OVERALL", "overall"],
    "target_role": ["target_role", "role", "nba_role"],
}


PROFILE_ALIASES = {
    "pts_per40": ["pts_per40", "pts", "ppg"],
    "ast_per40": ["ast_per40", "ast", "apg"],
    "reb_per40": ["reb_per40", "treb", "reb", "rpg"],
    "ts_pct": ["ts_pct", "TS_per", "ts"],
    "usg_pct": ["usg_pct", "usg", "USG_per"],
    "team_adj_o": ["team_adj_o", "ADJOE", "adj_o"],
    "team_adj_d": ["team_adj_d", "ADJDE", "adj_d"],
    "strength_of_schedule": ["strength_of_schedule", "sos", "SOS", "WAB"],
    "wingspan_in": ["wingspan_in", "WNGSPN", "wingspan"],
}


VECTOR_BASE_ALIASES = {
    "pts_per40": ["pts_per40", "ncaa_pts_per40", "pts", "ppg"],
    "ast_per40": ["ast_per40", "ncaa_ast_per40", "ast", "apg"],
    "reb_per40": ["reb_per40", "treb_per40", "ncaa_reb_per40", "treb", "reb", "rpg"],
    "stl_per40": ["stl_per40", "ncaa_stl_per40", "stl", "spg"],
    "blk_per40": ["blk_per40", "ncaa_blk_per40", "blk", "bpg"],
    "tov_per40": ["tov_per40", "ncaa_tov_per40", "tov", "to"],
    "ts_pct": ["ts_pct", "TS_per", "ts", "ncaa_ts"],
    "efg_pct": ["efg_pct", "eFG", "efg", "ncaa_efg"],
    "usg_pct": ["usg_pct", "usg", "USG_per", "ncaa_usg"],
    "ast_tov_ratio": ["ast_tov_ratio", "ast/tov"],
    "stocks_per40": ["stocks_per40"],
    "three_par": ["three_par", "3par", "TPA"],
    "ftr": ["ftr", "FTR", "ncaa_ftr"],
    "rec_rank": ["rec_rank", "Rec Rank", "recruiting_rank"],
    "is_unranked": ["is_unranked", "unranked"],
    "official_games": ["official_games", "bq_games"],
    "official_mpg": ["official_mpg", "bq_mpg"],
    "official_start_rate": ["official_start_rate", "bq_start_rate"],
    "height_in": ["height_in", "HGT", "height", "ht_in"],
    "weight_lb": ["weight_lb", "WGT", "weight", "wt"],
    "wingspan_in": ["wingspan_in", "WNGSPN", "wingspan"],
    "bmi": ["bmi", "BMI"],
    "wingspan_diff": ["wingspan_diff", "wingspan_minus_height"],
    "age_at_draft": ["age_at_draft", "age", "AGE"],
    "team_adj_o": ["team_adj_o", "ADJOE", "adj_o"],
    "team_adj_d": ["team_adj_d", "ADJDE", "adj_d"],
    "team_tempo": ["team_tempo", "ADJ_T", "tempo"],
    "team_power_rating": ["team_power_rating", "BARTHAG", "power_rating"],
    "strength_of_schedule": ["strength_of_schedule", "sos", "SOS", "team_sos", "WAB", "wab"],
    "draft_pick": ["draft_pick", "pick", "OVERALL", "overall"],
    "draft_year": ["draft_year", "year", "season"],
    "age": ["age", "AGE"],
}


def get_milvus_uri() -> str:
    """Build the Milvus URI from project configuration."""
    host = config.MILVUS_HOST
    if host.startswith(("http://", "https://")):
        return host
    return f"http://{host}:{config.MILVUS_PORT}"


def connect_to_milvus() -> MilvusClient:
    """Connect to Milvus using project configuration."""
    client = MilvusClient(uri=get_milvus_uri())
    print(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    return client


def create_collection(reset: bool = False) -> MilvusClient:
    """Create the project collection, optionally dropping an old version."""
    client = connect_to_milvus()

    if client.has_collection(config.MILVUS_COLLECTION_NAME):
        if reset:
            client.drop_collection(config.MILVUS_COLLECTION_NAME)
            print(f"Dropped existing collection '{config.MILVUS_COLLECTION_NAME}'")
        else:
            print(f"Collection '{config.MILVUS_COLLECTION_NAME}' already exists")
            return client

    schema = get_collection_schema()
    client.create_collection(
        collection_name=config.MILVUS_COLLECTION_NAME,
        schema=schema,
    )

    print(f"Created collection '{config.MILVUS_COLLECTION_NAME}'")
    return client


def deterministic_vector(seed: int, dim: int) -> list[float]:
    """Generate deterministic non-zero vectors for reproducible checkpoints."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def sample_player_rows() -> list[dict[str, object]]:
    """Return sample rows matching the Milvus schema."""
    players = [
        {
            "player_id": "stephen_curry_2009",
            "player_name": "Stephen Curry",
            "season": 2009,
            "college": "Davidson",
            "position": "G",
            "height_in": 75.0,
            "weight_lb": 185.0,
            "draft_year": 2009,
            "draft_pick": 7,
            "target_role": "STARTER_PLUS",
            "profile_text": (
                "High-usage guard from Davidson with elite shooting indicators, "
                "deep range, strong touch, and star NBA outcome."
            ),
        },
        {
            "player_id": "kawhi_leonard_2011",
            "player_name": "Kawhi Leonard",
            "season": 2011,
            "college": "San Diego State",
            "position": "F",
            "height_in": 79.0,
            "weight_lb": 225.0,
            "draft_year": 2011,
            "draft_pick": 15,
            "target_role": "STARTER_PLUS",
            "profile_text": (
                "Physical wing with elite defensive tools, rebounding, "
                "developmental shooting growth, and star NBA outcome."
            ),
        },
        {
            "player_id": "mikal_bridges_2018",
            "player_name": "Mikal Bridges",
            "season": 2018,
            "college": "Villanova",
            "position": "W",
            "height_in": 79.0,
            "weight_lb": 209.0,
            "draft_year": 2018,
            "draft_pick": 10,
            "target_role": "STARTER_PLUS",
            "profile_text": (
                "Long two-way wing from a strong Villanova context with "
                "shooting, defensive versatility, and starter NBA outcome."
            ),
        },
    ]

    for index, player in enumerate(players, start=1):
        player["numeric_vector"] = deterministic_vector(
            seed=100 + index,
            dim=config.NUMERIC_VECTOR_DIM,
        )
        player["text_vector"] = deterministic_vector(
            seed=200 + index,
            dim=config.EMBEDDING_DIM,
        )

    return players


def create_indexes(client: MilvusClient) -> None:
    """Create vector indexes for all vector fields."""
    index_params = client.prepare_index_params()
    for field_name, params in get_index_params().items():
        index_params.add_index(
            field_name=field_name,
            index_type=str(params["index_type"]),
            metric_type=str(params["metric_type"]),
            params=dict(params["params"]),
        )

    client.create_index(
        collection_name=config.MILVUS_COLLECTION_NAME,
        index_params=index_params,
    )
    print("Created indexes on 'numeric_vector' and 'text_vector'")


def get_entity_count(client: MilvusClient) -> int:
    """Return collection row count as an integer."""
    stats = client.get_collection_stats(config.MILVUS_COLLECTION_NAME)
    return int(stats.get("row_count", 0))


def _first_existing(row: pd.Series, aliases: list[str], default: Any = "") -> Any:
    """Return the first non-empty value found in a row."""
    for column in aliases:
        if column in row.index:
            value = row[column]
            if not pd.isna(value) and value != "":
                return value
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to an integer with a fallback."""
    return int(round(_safe_float(value, float(default))))


def _text_value(row: pd.Series, key: str, default: str = "unknown") -> str:
    """Return a string value using aliases for a semantic field."""
    value = _first_existing(row, FIELD_ALIASES.get(key, [key]), default=default)
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


def _numeric_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Return a numeric value using aliases for a semantic field."""
    value = _first_existing(row, FIELD_ALIASES.get(key, [key]), default=default)
    return _safe_float(value, default=default)


def _profile_numeric(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Return a numeric profile statistic using aliases."""
    value = _first_existing(row, PROFILE_ALIASES.get(key, [key]), default=default)
    return _safe_float(value, default=default)


def generate_profile_text(row: pd.Series | dict[str, Any]) -> str:
    """Generate a narrative text profile for semantic player search."""
    series = pd.Series(row)
    player_name = _text_value(series, "player_name", default="Unknown player")
    position = _text_value(series, "position", default="NA")
    college = _text_value(series, "college", default="unknown college")
    draft_year = _safe_int(_numeric_value(series, "draft_year"), default=0)

    height = _numeric_value(series, "height_in")
    weight = _numeric_value(series, "weight_lb")
    wingspan = _profile_numeric(series, "wingspan_in")
    pts = _profile_numeric(series, "pts_per40")
    ast = _profile_numeric(series, "ast_per40")
    reb = _profile_numeric(series, "reb_per40")
    ts_pct = _profile_numeric(series, "ts_pct")
    usage = _profile_numeric(series, "usg_pct")
    adj_o = _profile_numeric(series, "team_adj_o")
    adj_d = _profile_numeric(series, "team_adj_d")
    sos = _profile_numeric(series, "strength_of_schedule")

    draft_text = (
        f"draft year {draft_year}"
        if draft_year
        else "draft context unavailable"
    )
    physical_text = (
        f"{height:.0f} inches, {weight:.0f} lb, {wingspan:.0f} inch wingspan"
        if height or weight or wingspan
        else "physical measurements unavailable"
    )

    profile = (
        f"{player_name} is a {position} from {college}. "
        f"Draft context: {draft_text}. "
        f"NCAA profile: {pts:.1f} points per 40, {ast:.1f} assists per 40, "
        f"{reb:.1f} rebounds per 40, {ts_pct:.1f} true shooting, "
        f"{usage:.1f} usage. "
        f"Physical profile: {physical_text}. "
        f"Team context: adjusted offense {adj_o:.1f}, adjusted defense {adj_d:.1f}, "
        f"strength of schedule {sos:.1f}."
    )
    return profile[:1900]


def _z_score(series: pd.Series) -> pd.Series:
    """Return a finite z-score series."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = values.std(ddof=0)
    if not std or np.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _load_z_score_stats() -> dict[str, dict[str, float]]:
    """Load train-fit z-score stats when processed artifacts exist."""
    path = config.DATA_PROCESSED / "feature_scaler.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        stats = payload.get("stats", {})
    except (OSError, json.JSONDecodeError):
        return {}
    return stats if isinstance(stats, dict) else {}


def _first_numeric_column(df: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    """Return the first available numeric column from a frame."""
    for column in aliases:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return None


def _derived_base_series(df: pd.DataFrame, base_column: str) -> pd.Series | None:
    """Derive selected vector bases when raw component columns exist."""
    if base_column == "stocks_per40":
        steals = _first_numeric_column(df, VECTOR_BASE_ALIASES["stl_per40"])
        blocks = _first_numeric_column(df, VECTOR_BASE_ALIASES["blk_per40"])
        if steals is not None and blocks is not None:
            return steals.fillna(0.0) + blocks.fillna(0.0)

    if base_column == "ast_tov_ratio":
        assists = _first_numeric_column(df, VECTOR_BASE_ALIASES["ast_per40"])
        turnovers = _first_numeric_column(df, VECTOR_BASE_ALIASES["tov_per40"])
        if assists is not None and turnovers is not None:
            return (assists / turnovers.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if base_column == "bmi":
        weight = _first_numeric_column(df, VECTOR_BASE_ALIASES["weight_lb"])
        height = _first_numeric_column(df, VECTOR_BASE_ALIASES["height_in"])
        if weight is not None and height is not None:
            return ((weight * 703.0) / (height**2).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if base_column == "wingspan_diff":
        wingspan = _first_numeric_column(df, VECTOR_BASE_ALIASES["wingspan_in"])
        height = _first_numeric_column(df, VECTOR_BASE_ALIASES["height_in"])
        if wingspan is not None and height is not None:
            return wingspan - height

    if base_column == "is_unranked":
        rec_rank = _first_numeric_column(df, VECTOR_BASE_ALIASES["rec_rank"])
        if rec_rank is not None:
            return rec_rank.isna().astype(float)

    return None


def _column_or_derived_z(
    df: pd.DataFrame,
    column: str,
    z_score_stats: dict[str, dict[str, float]] | None = None,
) -> pd.Series:
    """Use an existing z-score column or derive it from a base column."""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")

    if column.endswith("_z"):
        base_column = column[:-2]
        base_series = None
        if base_column in df.columns:
            base_series = pd.to_numeric(df[base_column], errors="coerce")
        if base_series is None:
            base_series = _first_numeric_column(
                df,
                VECTOR_BASE_ALIASES.get(base_column, [base_column]),
            )
        if base_series is None:
            base_series = _derived_base_series(df, base_column)
        if base_series is not None:
            stats = (z_score_stats or {}).get(base_column, {})
            std = _safe_float(stats.get("std"), default=0.0)
            mean = _safe_float(stats.get("mean"), default=0.0)
            if std and not np.isclose(std, 0.0):
                values = pd.to_numeric(base_series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                return (values - mean) / std
            return _z_score(base_series)

    return pd.Series(0.0, index=df.index)


def normalize_vectors(
    batch: pd.DataFrame,
    z_score_stats: dict[str, dict[str, float]] | None = None,
) -> list[list[float]]:
    """Normalize numeric features to fixed-size vectors for Milvus."""
    if batch.empty:
        return []

    stats = z_score_stats if z_score_stats is not None else _load_z_score_stats()
    vector_frame = pd.DataFrame(index=batch.index)
    for column in NUMERIC_VECTOR_COLUMNS:
        vector_frame[column] = _column_or_derived_z(batch, column, stats)

    vector_frame = vector_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vectors = vector_frame.to_numpy(dtype=np.float32).copy()
    norms = np.linalg.norm(vectors, axis=1)

    for row_idx, norm in enumerate(norms):
        if np.isclose(norm, 0.0):
            vectors[row_idx, 0] = 1.0
        else:
            vectors[row_idx] = vectors[row_idx] / norm

    if vectors.shape[1] != config.NUMERIC_VECTOR_DIM:
        raise ValueError(
            "Numeric vector dimension mismatch: "
            f"{vectors.shape[1]} != {config.NUMERIC_VECTOR_DIM}"
        )

    return vectors.tolist()


def _hash_text_vector(text: str, dim: int = config.EMBEDDING_DIM) -> list[float]:
    """Build a deterministic text vector when transformer embeddings are unavailable."""
    values = []
    for index in range(dim):
        digest = hashlib.sha256(f"{text}|{index}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        values.append((integer / float(2**64 - 1)) * 2.0 - 1.0)

    vector = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm and not np.isclose(norm, 0.0):
        vector = vector / norm
    return vector.tolist()


def embed_profile_texts(texts: list[str]) -> list[list[float]]:
    """Embed profile texts with SentenceTransformers when explicitly enabled."""
    use_transformers = os.getenv("USE_SENTENCE_TRANSFORMERS", "0") == "1"
    if use_transformers:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.EMBEDDING_MODEL)
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            return embeddings.tolist()
        except Exception as exc:  # pragma: no cover - depends on local model cache/network
            print(
                "SentenceTransformer embeddings unavailable; "
                f"using deterministic text vectors instead. Reason: {exc}"
            )

    return [_hash_text_vector(text) for text in texts]


def make_demo_master_table() -> pd.DataFrame:
    """Create a richer deterministic table for full-ingest checkpoints."""
    rows = [
        ("stephen_curry_2009", "Stephen Curry", 2009, "Davidson", "G", 75, 185, 2009, 7, "STARTER_PLUS", 32.1, 6.7, 5.1, 63.0, 31.5, 118.2, 96.1, 3.4, 77),
        ("james_harden_2009", "James Harden", 2009, "Arizona State", "G", 77, 220, 2009, 3, "STARTER_PLUS", 25.2, 5.0, 7.2, 61.2, 28.6, 115.6, 93.5, 4.8, 80),
        ("kawhi_leonard_2011", "Kawhi Leonard", 2011, "San Diego State", "F", 79, 225, 2011, 15, "STARTER_PLUS", 18.9, 2.9, 12.4, 54.9, 24.2, 110.1, 88.4, 6.7, 87),
        ("mikal_bridges_2018", "Mikal Bridges", 2018, "Villanova", "W", 79, 209, 2018, 10, "STARTER_PLUS", 21.0, 2.8, 6.3, 65.0, 19.2, 127.8, 94.2, 10.1, 85),
        ("malcolm_brogdon_2016", "Malcolm Brogdon", 2016, "Virginia", "G", 77, 223, 2016, 36, "STARTER_PLUS", 21.4, 3.9, 5.0, 60.4, 24.1, 117.4, 86.3, 8.9, 82),
        ("draymond_green_2012", "Draymond Green", 2012, "Michigan State", "F", 78, 230, 2012, 35, "STARTER_PLUS", 18.5, 4.6, 12.0, 53.5, 26.4, 113.1, 89.7, 7.6, 85),
        ("desmond_bane_2020", "Desmond Bane", 2020, "TCU", "G", 77, 215, 2020, 30, "STARTER_PLUS", 20.4, 4.5, 7.1, 63.9, 23.5, 107.8, 96.4, 2.2, 78),
        ("jarrett_culver_2019", "Jarrett Culver", 2019, "Texas Tech", "W", 78, 195, 2019, 6, "BENCH", 21.7, 4.2, 7.8, 54.2, 30.1, 113.2, 85.1, 8.4, 81),
        ("tyus_jones_2015", "Tyus Jones", 2015, "Duke", "G", 73, 185, 2015, 24, "ROTATION", 15.6, 6.6, 4.2, 59.4, 20.5, 122.4, 92.8, 9.8, 75),
        ("frank_kaminsky_2015", "Frank Kaminsky", 2015, "Wisconsin", "C", 84, 242, 2015, 9, "ROTATION", 24.0, 3.0, 10.4, 62.6, 28.2, 126.0, 90.4, 11.2, 84),
        ("anthony_bennett_2013", "Anthony Bennett", 2013, "UNLV", "F", 80, 240, 2013, 1, "BENCH", 23.7, 1.5, 11.4, 61.1, 28.8, 108.5, 93.7, 1.5, 85),
        ("undrafted_guard_2018", "Demo Undrafted Guard", 2018, "Mid Major", "G", 72, 178, 0, 0, "NON_NBA", 12.8, 3.5, 3.0, 49.0, 18.5, 99.6, 105.1, -6.4, 73),
    ]
    columns = [
        "player_id",
        "player_name",
        "season",
        "college",
        "position",
        "height_in",
        "weight_lb",
        "draft_year",
        "draft_pick",
        "target_role",
        "pts_per40",
        "ast_per40",
        "reb_per40",
        "ts_pct",
        "usg_pct",
        "team_adj_o",
        "team_adj_d",
        "strength_of_schedule",
        "wingspan_in",
    ]
    data = pd.DataFrame(rows, columns=columns)
    data["stl_per40"] = [2.1, 1.8, 2.2, 1.9, 1.1, 1.7, 1.4, 1.8, 1.6, 0.7, 1.3, 0.8]
    data["blk_per40"] = [0.3, 0.6, 1.0, 1.4, 0.3, 1.3, 0.2, 0.8, 0.1, 2.0, 1.8, 0.1]
    data["tov_per40"] = [3.5, 3.7, 3.0, 1.9, 2.1, 3.6, 2.4, 4.1, 2.3, 2.2, 2.6, 2.7]
    data["efg_pct"] = [60.7, 58.0, 49.8, 63.6, 56.2, 49.5, 61.7, 48.5, 53.7, 58.0, 56.0, 45.0]
    data["ast_tov_ratio"] = data["ast_per40"] / data["tov_per40"].replace(0, np.nan)
    data["stocks_per40"] = data["stl_per40"] + data["blk_per40"]
    data["three_par"] = [0.52, 0.34, 0.18, 0.45, 0.38, 0.24, 0.57, 0.33, 0.41, 0.25, 0.23, 0.49]
    data["ftr"] = [38.0, 44.0, 42.0, 28.0, 39.0, 52.0, 20.0, 36.0, 41.0, 45.0, 54.0, 22.0]
    data["team_tempo"] = [69.4, 67.0, 66.1, 70.8, 61.8, 64.2, 67.5, 67.9, 66.2, 59.3, 68.1, 71.0]
    data["team_power_rating"] = [0.84, 0.87, 0.93, 0.98, 0.96, 0.91, 0.76, 0.95, 0.97, 0.98, 0.81, 0.32]
    data["age"] = [21.0, 19.8, 20.8, 21.6, 23.5, 22.3, 22.0, 20.4, 19.1, 22.2, 20.2, 22.1]
    data["age_at_draft"] = data["age"]
    data["rec_rank"] = [93, 98, 75, 88, 71, 67, 62, 89, 95, 91, 99, 0]
    data["is_unranked"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    data["official_games"] = [34, 35, 34, 40, 34, 37, 32, 38, 39, 40, 35, 30]
    data["official_mpg"] = [33.0, 35.0, 31.0, 32.0, 34.0, 33.0, 36.0, 32.0, 29.0, 34.0, 28.0, 20.0]
    data["official_start_rate"] = [1.0, 1.0, 0.95, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 1.0, 0.85, 0.1]
    data["bmi"] = (data["weight_lb"] / (data["height_in"] ** 2)) * 703.0
    data["wingspan_diff"] = data["wingspan_in"] - data["height_in"]
    return data


def load_master_table_or_demo() -> tuple[pd.DataFrame, str]:
    """Load the processed master table when available, otherwise demo data."""
    master_path = config.DATA_PROCESSED / "master_table.csv"
    if master_path.exists():
        master = pd.read_csv(master_path, low_memory=False)
        return master, str(master_path)

    return make_demo_master_table(), "deterministic demo table"


def _build_player_id(row: pd.Series, index: int) -> str:
    """Create a stable player id when the table does not already contain one."""
    existing = _text_value(row, "player_id", default="")
    if existing:
        return existing[:256]

    name = _text_value(row, "player_name", default=f"player_{index}")
    season = _safe_int(_numeric_value(row, "season"), default=0)
    slug = "".join(char.lower() if char.isalnum() else "_" for char in name)
    slug = "_".join(part for part in slug.split("_") if part)
    return f"{slug}_{season or index}"[:256]


def _prepare_milvus_rows(master_df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a master table to Milvus row dictionaries."""
    normalized_master = master_df.copy()
    numeric_vectors = normalize_vectors(normalized_master)
    profile_texts = [
        generate_profile_text(row)
        for _, row in normalized_master.iterrows()
    ]
    text_vectors = embed_profile_texts(profile_texts)

    rows = []
    for index, (_, row) in enumerate(normalized_master.iterrows()):
        rows.append(
            {
                "player_id": _build_player_id(row, index),
                "player_name": _text_value(row, "player_name", default=f"Player {index}"),
                "season": _safe_int(_numeric_value(row, "season"), default=0),
                "college": _text_value(row, "college", default="unknown"),
                "position": _text_value(row, "position", default="NA")[:10],
                "height_in": _numeric_value(row, "height_in"),
                "weight_lb": _numeric_value(row, "weight_lb"),
                "draft_year": _safe_int(_numeric_value(row, "draft_year"), default=0),
                "draft_pick": _safe_int(_numeric_value(row, "draft_pick"), default=0),
                "target_role": _text_value(row, "target_role", default="UNKNOWN")[:20],
                "numeric_vector": numeric_vectors[index],
                "text_vector": text_vectors[index],
                "profile_text": profile_texts[index],
            }
        )

    return rows


def ingest_sample_data(reset: bool = True) -> dict[str, object]:
    """Create the collection and ingest deterministic sample players."""
    client = create_collection(reset=reset)
    sample_data = sample_player_rows()

    client.insert(collection_name=config.MILVUS_COLLECTION_NAME, data=sample_data)
    client.flush(config.MILVUS_COLLECTION_NAME)

    inserted = len(sample_data)
    print(f"Inserted {inserted} sample player records")

    create_indexes(client)
    client.load_collection(config.MILVUS_COLLECTION_NAME)
    entity_count = get_entity_count(client)
    print(f"Collection loaded. Total entities: {entity_count}")

    return {
        "collection_name": config.MILVUS_COLLECTION_NAME,
        "entities": entity_count,
    }


def ingest_full_dataset(
    master_df: pd.DataFrame,
    reset: bool = True,
    batch_size: int = 100,
) -> dict[str, object]:
    """Create the collection and ingest a complete player table."""
    if master_df.empty:
        raise ValueError("Cannot ingest an empty master table")

    started = time.perf_counter()
    client = create_collection(reset=reset)
    milvus_rows = _prepare_milvus_rows(master_df)

    inserted = 0
    for start in range(0, len(milvus_rows), batch_size):
        batch = milvus_rows[start : start + batch_size]
        client.insert(collection_name=config.MILVUS_COLLECTION_NAME, data=batch)
        inserted += len(batch)
        print(f"Inserted {inserted} / {len(milvus_rows)} player records")

    client.flush(config.MILVUS_COLLECTION_NAME)
    create_indexes(client)
    client.load_collection(config.MILVUS_COLLECTION_NAME)

    entity_count = get_entity_count(client)
    elapsed = time.perf_counter() - started
    print(f"Collection loaded. Total entities: {entity_count}")
    print(f"Full ingestion elapsed time: {elapsed:.2f}s")

    return {
        "collection_name": config.MILVUS_COLLECTION_NAME,
        "entities": entity_count,
        "inserted": inserted,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    """Run the Milvus sample-ingestion checkpoint."""
    result = ingest_sample_data(reset=True)
    print("\nMilvus ingestion checkpoint completed.")
    print(f"Collection: {result['collection_name']}")
    print(f"Entities: {result['entities']}")


def main_full() -> None:
    """Run the full Milvus ingestion checkpoint."""
    master, source = load_master_table_or_demo()
    print(f"Loaded source: {source}")
    print(f"Rows to ingest: {len(master)}")

    result = ingest_full_dataset(master, reset=True)
    print("\nMilvus full ingestion checkpoint completed.")
    print(f"Collection: {result['collection_name']}")
    print(f"Inserted: {result['inserted']}")
    print(f"Entities: {result['entities']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    main()
