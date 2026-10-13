"""Streamlit demo for the NCAA to NBA role predictor."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st


ROLE_ORDER = ["NON_NBA", "NBA_ROLE_PLAYER", "STARTER_PLUS"]
DEFAULT_INDEXED_PLAYER = "Joel Embiid"
CURRENT_SEASON_YEAR = 2026
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAYER_DATA_DIR = PROJECT_ROOT / "Datasets" / "College Basketball 2009-2021 + NBA Advanced Stats"
TEAM_CONTEXT_DIR = PROJECT_ROOT / "Datasets" / "College Basketball Dataset"
BART_TORVIK_DIR = PROJECT_ROOT / "Datasets" / "Bart Torvik Data"
CURRENT_PLAYER_CANDIDATES = [
    BART_TORVIK_DIR / "BartTorvik2026.csv",
    PLAYER_DATA_DIR / "CollegeBasketballPlayers2026.csv",
    PROJECT_ROOT / "Datasets" / "CollegeBasketballPlayers2026.csv",
    PROJECT_ROOT / "Datasets" / "CollegeBasketballPlayers2025-2026.csv",
    PROJECT_ROOT / "data" / "processed" / "current_ncaa_players.csv",
]
MONTH_HEIGHTS = {"May": 5, "Jun": 6, "Jul": 7}
BART_TORVIK_PLAYER_COLUMNS = [
    "player_name",
    "team",
    "conf",
    "GP",
    "Min_per",
    "Ortg",
    "usg",
    "eFG",
    "TS_per",
    "ORB_per",
    "DRB_per",
    "AST_per",
    "TO_per",
    "FTM",
    "FTA",
    "FT_per",
    "twoPM",
    "twoPA",
    "twoP_per",
    "TPM",
    "TPA",
    "TP_per",
    "blk_per",
    "stl_per",
    "ftr",
    "yr",
    "ht",
    "num",
    "porpag",
    "adjoe",
    "pfr",
    "year",
    "pid",
    "hometown",
    "Rec Rank",
    "ast/tov",
    "rimmade",
    "rimmade+rimmiss",
    "midmade",
    "midmade+midmiss",
    "rimmade/(rimmade+rimmiss)",
    "midmade/(midmade+midmiss)",
    "dunksmade",
    "dunksmiss+dunksmade",
    "dunksmade/(dunksmade+dunksmiss)",
    "pick",
    "drtg",
    "adrtg",
    "dporpag",
    "stops",
    "bpm",
    "obpm",
    "dbpm",
    "gbpm",
    "mp",
    "ogbpm",
    "dgbpm",
    "oreb",
    "dreb",
    "treb",
    "ast",
    "stl",
    "blk",
    "pts",
    "position_label",
    "source_score",
    "birthdate",
]


def get_api_base_url() -> str:
    """Return the API base URL for local or Docker execution."""
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def call_api(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the FastAPI service and return a JSON payload or an error."""
    url = f"{get_api_base_url()}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            json=json_body,
            params=params,
            timeout=timeout,
        )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            return None, f"{response.status_code}: {detail}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def normalize_text(value: Any) -> str:
    """Normalize names for simple matching."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def as_float(value: Any, default: float = 0.0) -> float:
    """Return a safe float for CSV values."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return default
    return float(number)


def optional_float(value: Any) -> float | None:
    """Return a float when present, otherwise None for API optional fields."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def parse_height_inches(value: Any, default: float = 76.0) -> float:
    """Parse common NCAA height strings, including Excel-damaged values."""
    if pd.isna(value):
        return default

    text = str(value).strip()
    if not text:
        return default

    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        number = float(numeric)
        return number if number > 30 else default

    feet_inches = re.match(r"^(\d+)\s*[-']\s*(\d+)", text)
    if feet_inches:
        feet = int(feet_inches.group(1))
        inches = int(feet_inches.group(2))
        if 4 <= feet <= 7 and 0 <= inches < 12:
            return float(feet * 12 + inches)

    parts = text.split("-")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        if first in MONTH_HEIGHTS:
            inches = int(second) if second.isdigit() else 0
            return float(MONTH_HEIGHTS[first] * 12 + inches)
        if second in MONTH_HEIGHTS and first.isdigit():
            return float(MONTH_HEIGHTS[second] * 12 + int(first))

    return default


def simplify_position(value: Any) -> str:
    """Map source position labels to API position buckets."""
    text = str(value or "").upper()
    if "C" in text and "PG" not in text:
        return "C"
    if "PF" in text or "STRETCH" in text or "F" in text:
        return "F"
    if "WING" in text:
        return "W"
    return "G"


def default_weight(position: str) -> float:
    """Return a plausible default weight when the current source lacks it."""
    return {"G": 190.0, "W": 205.0, "F": 220.0, "C": 240.0}.get(position, 205.0)


def wingspan_from_height(height_in: float, position: str) -> float:
    """Return a conservative wingspan estimate when combine data is unavailable."""
    bonus = {"G": 2.5, "W": 3.5, "F": 4.5, "C": 5.0}.get(position, 3.0)
    return float(height_in + bonus)


def per40(row: pd.Series, column: str, minutes_column: str = "mp") -> float:
    """Convert a per-game stat to per-40 minutes."""
    minutes = as_float(row.get(minutes_column), default=0.0)
    value = as_float(row.get(column), default=0.0)
    if minutes <= 0:
        return 0.0
    return float(value * 40.0 / minutes)


def source_year_from_path(path: Path) -> int:
    """Infer the season year from a player source filename."""
    match = re.search(r"(20\d{2})", path.name)
    if match:
        return int(match.group(1))
    return CURRENT_SEASON_YEAR


def is_bart_torvik_player_file(path: Path) -> bool:
    """Return whether a path points to a raw Bart Torvik player export."""
    return path.name.startswith("BartTorvik")


def read_player_source(path: Path) -> pd.DataFrame:
    """Read a player-level source, handling Bart Torvik's headerless exports."""
    if is_bart_torvik_player_file(path):
        return pd.read_csv(path, header=None, names=BART_TORVIK_PLAYER_COLUMNS, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def locate_player_source() -> tuple[Path | None, int | None, bool]:
    """Find the current player table or the latest local player table."""
    for path in CURRENT_PLAYER_CANDIDATES:
        if path.exists():
            return path, CURRENT_SEASON_YEAR, True

    candidates = sorted(PLAYER_DATA_DIR.glob("CollegeBasketballPlayers*.csv"))
    if not candidates:
        return None, None, False

    latest = max(candidates, key=source_year_from_path)
    return latest, source_year_from_path(latest), False


@st.cache_data(show_spinner=False)
def load_team_context(year: int) -> pd.DataFrame:
    """Load team context for the selected season."""
    bart_path = BART_TORVIK_DIR / f"{year}_team_results.csv"
    if bart_path.exists():
        context = pd.read_csv(bart_path, low_memory=False)
        context["_team_key"] = context["team"].map(normalize_text)
        return pd.DataFrame(
            {
                "_team_key": context["_team_key"],
                "ADJOE": pd.to_numeric(context.get("adjoe"), errors="coerce"),
                "ADJDE": pd.to_numeric(context.get("adjde"), errors="coerce"),
                "ADJ_T": pd.to_numeric(context.get("adjt"), errors="coerce"),
                "BARTHAG": pd.to_numeric(context.get("barthag"), errors="coerce"),
                "WAB": pd.to_numeric(context.get("WAB"), errors="coerce"),
                "SOS": pd.to_numeric(context.get("sos"), errors="coerce"),
            }
        )

    suffix = str(year)[-2:]
    path = TEAM_CONTEXT_DIR / f"cbb{suffix}.csv"
    if not path.exists() and year == 2026:
        path = TEAM_CONTEXT_DIR / "cbb26.csv"
    if not path.exists():
        return pd.DataFrame()

    context = pd.read_csv(path)
    context["_team_key"] = context["TEAM"].map(normalize_text)
    return pd.DataFrame(
        {
            "_team_key": context["_team_key"],
            "ADJOE": pd.to_numeric(context.get("ADJOE"), errors="coerce"),
            "ADJDE": pd.to_numeric(context.get("ADJDE"), errors="coerce"),
            "ADJ_T": pd.to_numeric(context.get("ADJ_T"), errors="coerce"),
            "BARTHAG": pd.to_numeric(context.get("BARTHAG"), errors="coerce"),
            "WAB": pd.to_numeric(context.get("WAB"), errors="coerce"),
            "SOS": pd.Series(0.0, index=context.index),
        }
    )


@st.cache_data(show_spinner=False)
def load_player_catalog() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the best available player-level NCAA table for the selector."""
    source, source_year, is_current = locate_player_source()
    if source is None or source_year is None:
        return pd.DataFrame(), {
            "source_path": None,
            "source_year": None,
            "is_current": False,
            "message": "No se encontro una tabla individual de jugadores NCAA.",
        }

    players = read_player_source(source)
    if "player_name" not in players.columns:
        return pd.DataFrame(), {
            "source_path": str(source),
            "source_year": source_year,
            "is_current": is_current,
            "message": "La tabla encontrada no contiene la columna player_name.",
        }

    players = players[players["player_name"].notna()].copy()
    players["_source_year"] = source_year
    players["_is_current"] = is_current
    players["_team_key"] = players.get("team", "").map(normalize_text)

    context = load_team_context(source_year)
    if not context.empty:
        players = players.merge(
            context[["_team_key", "ADJOE", "ADJDE", "ADJ_T", "BARTHAG", "WAB", "SOS"]],
            how="left",
            on="_team_key",
            suffixes=("", "_team"),
        )

    players["_search_name"] = players["player_name"].map(normalize_text)
    players["_option_label"] = players.apply(player_option_label, axis=1)
    return players, {
        "source_path": str(source.relative_to(PROJECT_ROOT)),
        "source_year": source_year,
        "is_current": is_current,
        "message": "",
    }


def player_option_label(row: pd.Series) -> str:
    """Build a readable selector label."""
    name = str(row.get("player_name", "Jugador"))
    team = str(row.get("team", "Equipo"))
    year = str(row.get("yr", "") or "").strip()
    points = as_float(row.get("pts"), default=0.0)
    return f"{name} | {team} | {year} | {points:.1f} PPG"


def filter_catalog(catalog: pd.DataFrame, query: str, limit: int = 200) -> pd.DataFrame:
    """Filter catalog rows by user query."""
    if catalog.empty:
        return catalog
    normalized_query = normalize_text(query)
    if not normalized_query:
        filtered = catalog.sort_values(["pts", "mp"], ascending=False, na_position="last")
    else:
        mask = catalog["_search_name"].str.contains(normalized_query, na=False)
        filtered = catalog[mask]
    return filtered.head(limit)


def select_catalog_row(catalog: pd.DataFrame, key: str, label: str = "Jugador") -> pd.Series:
    """Render one searchable selector and return the selected catalog row."""
    ordered = catalog.sort_values(["pts", "mp"], ascending=False, na_position="last").reset_index(drop=True)
    selected_index = st.selectbox(
        label,
        options=list(ordered.index),
        format_func=lambda index: str(ordered.loc[index, "_option_label"]),
        key=key,
    )
    return ordered.loc[int(selected_index)]


def age_from_class(value: Any) -> float:
    """Approximate age from NCAA class when explicit age is unavailable."""
    text = str(value or "").lower()
    if text.startswith("fr"):
        return 19.0
    if text.startswith("so"):
        return 20.0
    if text.startswith("jr"):
        return 21.0
    if text.startswith("sr"):
        return 22.0
    return 20.5


def age_from_birthdate(value: Any, reference_year: int) -> float | None:
    """Calculate age around draft time when Bart Torvik includes birthdate."""
    birthdate = pd.to_datetime(value, errors="coerce")
    if pd.isna(birthdate):
        return None
    reference_date = pd.Timestamp(year=int(reference_year), month=6, day=30)
    age = (reference_date - birthdate).days / 365.25
    if 16 <= age <= 30:
        return float(round(age, 2))
    return None


def selected_player_payload(row: pd.Series) -> dict[str, Any]:
    """Convert a loaded NCAA player row into the API request schema."""
    source_year = int(row.get("_source_year") or CURRENT_SEASON_YEAR)
    position = simplify_position(row.get("position_label", row.get("Unnamed: 64")))
    height_in = parse_height_inches(row.get("ht"))
    ast_per40 = per40(row, "ast")
    ast_tov_ratio = as_float(row.get("ast/tov"), default=0.0)
    tov_per40 = ast_per40 / ast_tov_ratio if ast_tov_ratio > 0 else 2.5
    three_attempts = as_float(row.get("TPA"), default=0.0)
    two_attempts = as_float(row.get("twoPA"), default=0.0)
    shot_attempts = three_attempts + two_attempts
    age = age_from_birthdate(row.get("birthdate"), source_year) or age_from_class(row.get("yr"))
    rec_rank = optional_float(row.get("Rec Rank"))
    games = as_float(row.get("GP"), default=0.0)
    minutes = as_float(row.get("mp"), default=0.0)
    minutes_share = as_float(row.get("Min_per"), default=0.0)

    return {
        "player_name": str(row.get("player_name", "Prospect")),
        "position": position,
        "college": str(row.get("team", "Unknown")),
        "season": source_year,
        "draft_year": source_year,
        "draft_pick": 0,
        "age": age,
        "age_at_draft": age,
        "rec_rank": rec_rank,
        "is_unranked": 1.0 if rec_rank is None else 0.0,
        "official_games": games,
        "official_mpg": minutes,
        "official_start_rate": max(0.0, min(1.0, minutes_share / 100.0)),
        "height_in": height_in,
        "weight_lb": default_weight(position),
        "wingspan_in": wingspan_from_height(height_in, position),
        "pts_per40": per40(row, "pts"),
        "ast_per40": ast_per40,
        "reb_per40": per40(row, "treb"),
        "stl_per40": per40(row, "stl"),
        "blk_per40": per40(row, "blk"),
        "tov_per40": float(tov_per40),
        "ts_pct": as_float(row.get("TS_per"), default=55.0),
        "efg_pct": as_float(row.get("eFG"), default=51.0),
        "usg_pct": as_float(row.get("usg"), default=20.0),
        "ast_tov_ratio": ast_tov_ratio if ast_tov_ratio > 0 else None,
        "three_par": three_attempts / shot_attempts if shot_attempts > 0 else 0.35,
        "ftr": as_float(row.get("ftr"), default=30.0),
        "team_adj_o": as_float(row.get("ADJOE"), default=as_float(row.get("adjoe"), default=110.0)),
        "team_adj_d": as_float(row.get("ADJDE"), default=100.0),
        "team_tempo": as_float(row.get("ADJ_T"), default=68.0),
        "team_power_rating": as_float(row.get("BARTHAG"), default=0.70),
        "strength_of_schedule": as_float(row.get("SOS"), default=as_float(row.get("WAB"), default=0.0)),
    }


def selected_player_summary(row: pd.Series) -> pd.DataFrame:
    """Return a compact visible summary for a selected catalog player."""
    payload = selected_player_payload(row)
    summary = pd.DataFrame(
        [
            {"Campo": "Jugador", "Valor": payload["player_name"]},
            {"Campo": "Universidad", "Valor": payload["college"]},
            {"Campo": "Temporada fuente", "Valor": payload["season"]},
            {"Campo": "Posicion", "Valor": payload["position"]},
            {"Campo": "PTS/40", "Valor": round(payload["pts_per40"], 2)},
            {"Campo": "TS%", "Valor": round(payload["ts_pct"], 2)},
            {"Campo": "USG%", "Valor": round(payload["usg_pct"], 2)},
            {"Campo": "Rec Rank", "Valor": payload.get("rec_rank") if payload.get("rec_rank") is not None else "Sin ranking"},
            {"Campo": "Minutos", "Valor": round(payload.get("official_mpg") or 0.0, 2)},
        ]
    )
    summary["Valor"] = summary["Valor"].astype(str)
    return summary


def manual_payload(values: dict[str, Any]) -> dict[str, Any]:
    """Build an API payload from manual inputs."""
    return {
        "player_name": str(values["player_name"]),
        "position": str(values["position"]),
        "college": str(values["college"]),
        "season": int(values["cohort_year"]),
        "draft_year": int(values["cohort_year"]),
        "draft_pick": 0,
        "age": float(values["age"]),
        "age_at_draft": float(values["age"]),
        "rec_rank": values.get("rec_rank"),
        "is_unranked": float(values.get("is_unranked", 0.0)),
        "official_games": float(values.get("official_games", 0.0)),
        "official_mpg": float(values.get("official_mpg", 0.0)),
        "official_start_rate": float(values.get("official_start_rate", 0.0)),
        "height_in": float(values["height_in"]),
        "weight_lb": float(values["weight_lb"]),
        "wingspan_in": float(values["wingspan_in"]),
        "pts_per40": float(values["pts_per40"]),
        "ast_per40": float(values["ast_per40"]),
        "reb_per40": float(values["reb_per40"]),
        "stl_per40": float(values["stl_per40"]),
        "blk_per40": float(values["blk_per40"]),
        "tov_per40": float(values["tov_per40"]),
        "ts_pct": float(values["ts_pct"]),
        "efg_pct": float(values["efg_pct"]),
        "usg_pct": float(values["usg_pct"]),
        "three_par": float(values["three_par"]),
        "ftr": float(values["ftr"]),
        "team_adj_o": float(values["team_adj_o"]),
        "team_adj_d": float(values["team_adj_d"]),
        "team_tempo": float(values["team_tempo"]),
        "team_power_rating": float(values["team_power_rating"]),
        "strength_of_schedule": float(values["strength_of_schedule"]),
    }


def submit_prediction(payload: dict[str, Any]) -> None:
    """Call the API and store the prediction in session state."""
    prediction, error = call_api("POST", "/predict-role", json_body=payload)
    if error:
        st.session_state["prediction_error"] = error
        st.session_state.pop("prediction", None)
    else:
        st.session_state["prediction"] = prediction
        st.session_state.pop("prediction_error", None)


def probability_frame(probabilities: dict[str, float]) -> pd.DataFrame:
    """Format prediction probabilities in project role order."""
    rows = [
        {
            "role": role,
            "probability": float(probabilities.get(role, 0.0)),
        }
        for role in ROLE_ORDER
    ]
    return pd.DataFrame(rows)


def render_probability_bars(probabilities: dict[str, float]) -> None:
    """Render probability bars with stable ordering."""
    frame = probability_frame(probabilities)
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("role:N", sort=ROLE_ORDER, title=None),
            y=alt.Y("probability:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("role:N", title="Rol"),
                alt.Tooltip("probability:Q", title="Probabilidad", format=".1%"),
            ],
            color=alt.Color("role:N", sort=ROLE_ORDER, legend=None),
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def comparable_frame(comparables: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a compact table for comparable players."""
    if not comparables:
        return pd.DataFrame(columns=["Jugador", "Rol", "Universidad", "Draft", "Score"])

    frame = pd.DataFrame(comparables)
    frame = frame.rename(
        columns={
            "player_name": "Jugador",
            "target_role": "Rol",
            "college": "Universidad",
            "draft_year": "Draft",
            "score": "Score",
        }
    )
    columns = ["Jugador", "Rol", "Universidad", "Draft", "Score"]
    frame = frame.reindex(columns=columns)
    if "Score" in frame.columns:
        frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce").round(4)
    return frame


def render_status() -> None:
    """Show API status in a compact sidebar block."""
    health, error = call_api("GET", "/", timeout=5.0)
    if error:
        st.sidebar.error("API no disponible")
        st.sidebar.caption(error)
        return

    model_name = health.get("model_name") or "sin modelo"
    collection = health.get("milvus_collection") or "sin coleccion"
    st.sidebar.success("API activa")
    st.sidebar.caption(f"Modelo: {model_name}")
    st.sidebar.caption(f"Milvus: {collection}")


def render_loaded_player_mode() -> None:
    """Render selector-based prediction mode."""
    catalog, meta = load_player_catalog()
    if catalog.empty:
        st.warning(meta.get("message") or "No hay jugadores cargados para seleccionar.")
        return

    if meta["is_current"]:
        st.success(f"Selector usando jugadores 2025/26: `{meta['source_path']}`")
    else:
        st.warning(
            "No hay una tabla individual local de jugadores 2025/26. "
            f"Usando la ultima tabla individual disponible: `{meta['source_path']}`."
        )

    selected = select_catalog_row(catalog, key="prediction_player_selector", label="Buscar jugador")

    st.dataframe(selected_player_summary(selected), width="stretch", hide_index=True)
    if st.button("Predecir jugador seleccionado", width="stretch"):
        submit_prediction(selected_player_payload(selected))


def render_manual_mode() -> None:
    """Render manual input mode."""
    with st.form("prediction_form"):
        identity_left, identity_right = st.columns(2)
        with identity_left:
            player_name = st.text_input("Jugador", value="Prospect Demo")
            college = st.text_input("Universidad", value="Kansas")
            position = st.selectbox("Posicion", ["G", "W", "F", "C"], index=0)
        with identity_right:
            cohort_year = st.number_input("Anio de referencia", min_value=2009, max_value=2026, value=2026)
            age = st.slider("Edad", min_value=17.0, max_value=25.0, value=20.5, step=0.1)
            is_unranked = st.checkbox("Sin ranking recruiting", value=False)
            rec_rank = st.slider("Rec Rank", min_value=0.0, max_value=100.0, value=50.0, step=1.0, disabled=is_unranked)

        physical_left, physical_right, physical_third = st.columns(3)
        with physical_left:
            height_in = st.slider("Altura", min_value=68.0, max_value=86.0, value=76.0, step=0.5)
        with physical_right:
            weight_lb = st.slider("Peso", min_value=150.0, max_value=300.0, value=205.0, step=1.0)
        with physical_third:
            wingspan_in = st.slider("Envergadura", min_value=68.0, max_value=92.0, value=79.0, step=0.5)

        stats_left, stats_right, stats_third = st.columns(3)
        with stats_left:
            pts_per40 = st.slider("PTS/40", min_value=0.0, max_value=40.0, value=21.5, step=0.1)
            stl_per40 = st.slider("STL/40", min_value=0.0, max_value=5.0, value=1.4, step=0.1)
        with stats_right:
            ast_per40 = st.slider("AST/40", min_value=0.0, max_value=15.0, value=4.8, step=0.1)
            blk_per40 = st.slider("BLK/40", min_value=0.0, max_value=6.0, value=0.6, step=0.1)
        with stats_third:
            reb_per40 = st.slider("REB/40", min_value=0.0, max_value=20.0, value=6.4, step=0.1)
            tov_per40 = st.slider("TOV/40", min_value=0.0, max_value=8.0, value=2.6, step=0.1)

        eff_left, eff_right, eff_third = st.columns(3)
        with eff_left:
            ts_pct = st.slider("TS%", min_value=35.0, max_value=75.0, value=59.0, step=0.1)
            three_par = st.slider("3PAr", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        with eff_right:
            efg_pct = st.slider("eFG%", min_value=30.0, max_value=75.0, value=54.0, step=0.1)
            ftr = st.slider("FTr", min_value=0.0, max_value=80.0, value=32.0, step=0.5)
        with eff_third:
            usg_pct = st.slider("USG%", min_value=5.0, max_value=40.0, value=24.5, step=0.1)

        context_left, context_right, context_third = st.columns(3)
        with context_left:
            team_adj_o = st.slider("AdjO", min_value=80.0, max_value=130.0, value=116.0, step=0.1)
            team_power_rating = st.slider("Power", min_value=0.0, max_value=1.0, value=0.78, step=0.01)
        with context_right:
            team_adj_d = st.slider("AdjD", min_value=80.0, max_value=120.0, value=96.0, step=0.1)
            strength_of_schedule = st.slider("SoS", min_value=-20.0, max_value=20.0, value=2.0, step=0.1)
        with context_third:
            team_tempo = st.slider("Tempo", min_value=55.0, max_value=85.0, value=68.0, step=0.1)

        role_left, role_right, role_third = st.columns(3)
        with role_left:
            official_games = st.slider("Partidos", min_value=0.0, max_value=40.0, value=30.0, step=1.0)
        with role_right:
            official_mpg = st.slider("Minutos", min_value=0.0, max_value=40.0, value=28.0, step=0.1)
        with role_third:
            official_start_rate = st.slider("Titularidad", min_value=0.0, max_value=1.0, value=0.70, step=0.01)

        submitted = st.form_submit_button("Predecir rol", width="stretch")

    if submitted:
        submit_prediction(
            manual_payload(
                {
                    "player_name": player_name,
                    "position": position,
                    "college": college,
                    "cohort_year": cohort_year,
                    "age": age,
                    "rec_rank": None if is_unranked else rec_rank,
                    "is_unranked": 1.0 if is_unranked else 0.0,
                    "official_games": official_games,
                    "official_mpg": official_mpg,
                    "official_start_rate": official_start_rate,
                    "height_in": height_in,
                    "weight_lb": weight_lb,
                    "wingspan_in": wingspan_in,
                    "pts_per40": pts_per40,
                    "ast_per40": ast_per40,
                    "reb_per40": reb_per40,
                    "stl_per40": stl_per40,
                    "blk_per40": blk_per40,
                    "tov_per40": tov_per40,
                    "ts_pct": ts_pct,
                    "efg_pct": efg_pct,
                    "usg_pct": usg_pct,
                    "three_par": three_par,
                    "ftr": ftr,
                    "team_adj_o": team_adj_o,
                    "team_adj_d": team_adj_d,
                    "team_tempo": team_tempo,
                    "team_power_rating": team_power_rating,
                    "strength_of_schedule": strength_of_schedule,
                }
            )
        )


def render_prediction_panel() -> None:
    """Render the latest prediction result."""
    st.subheader("Prediccion")
    prediction_error = st.session_state.get("prediction_error")
    prediction = st.session_state.get("prediction")

    if prediction_error:
        st.error(prediction_error)
    elif prediction:
        role = prediction["predicted_role"]
        confidence = float(prediction["confidence"])
        st.markdown(f'<span class="role-pill">{role}</span>', unsafe_allow_html=True)
        st.metric("Confianza", f"{confidence:.1%}")
        render_probability_bars(prediction.get("probabilities", {}))
        st.caption(f"Modelo: {prediction.get('model_name')} | Fuente: {prediction.get('feature_source')}")
    else:
        st.info("Sin prediccion activa.")


def render_comparables_panel() -> None:
    """Render comparable-player lookup."""
    st.subheader("Comparables")
    catalog, meta = load_player_catalog()
    if catalog.empty:
        st.warning(meta.get("message") or "No hay jugadores NCAA actuales para comparar.")
        return

    selected = select_catalog_row(catalog, key="comparable_player_selector", label="Buscar jugador NCAA")
    st.dataframe(selected_player_summary(selected), width="stretch", hide_index=True)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, key="comparables_top_k")
    if st.button("Buscar comparables", width="stretch"):
        comparables, error = call_api(
            "POST",
            "/comparable-players/profile",
            json_body=selected_player_payload(selected),
            params={"top_k": int(top_k)},
        )
        if error:
            st.session_state["comparables_error"] = error
            st.session_state.pop("comparables", None)
        else:
            st.session_state["comparables"] = comparables
            st.session_state.pop("comparables_error", None)


def render_prediction_mode() -> None:
    """Render the role-prediction workflow."""
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Perfil NCAA")
        customize = st.toggle("Personalizar datos", value=False)
        if customize:
            render_manual_mode()
        else:
            render_loaded_player_mode()

    with right:
        render_prediction_panel()


def render_similarity_mode() -> None:
    """Render the profile-similarity workflow."""
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        render_comparables_panel()

    with right:
        comparables_error = st.session_state.get("comparables_error")
        comparables = st.session_state.get("comparables")
        st.subheader("Resultado")
        if comparables_error:
            st.warning(comparables_error)
        elif comparables:
            st.dataframe(
                comparable_frame(comparables.get("comparables", [])),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Selecciona un jugador NCAA y busca sus comparables historicos.")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="NCAA to NBA Role Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            max-width: 1240px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }
        .role-pill {
            display: inline-block;
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            background: #0f766e;
            color: white;
            font-weight: 700;
            letter-spacing: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Servicio")
    st.sidebar.text_input("API", value=get_api_base_url(), disabled=True)
    render_status()

    st.title("NCAA to NBA Role Predictor")

    mode = st.radio(
        "Modo",
        ["Prediccion", "Similitud"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Prediccion":
        render_prediction_mode()
    else:
        render_similarity_mode()


if __name__ == "__main__":
    main()
