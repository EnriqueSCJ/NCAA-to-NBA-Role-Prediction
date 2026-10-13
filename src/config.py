"""Project-wide configuration.

The values in this module are intentionally simple constants so every module can
import the same paths, model settings, and Milvus settings without duplicating
project knowledge.
"""

from __future__ import annotations

import os
from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Source dataset paths already present in this workspace.
NCAA_NBA_DATASET_DIR = DATASETS_DIR / "College Basketball 2009-2021 + NBA Advanced Stats"
COLLEGE_CONTEXT_DATASET_DIR = DATASETS_DIR / "College Basketball Dataset"
BART_TORVIK_DATASET_DIR = DATASETS_DIR / "Bart Torvik Data"
NCAA_BIGQUERY_EXPORT_DIR = DATASETS_DIR / "NCAA BigQuery Export"
BASKETBALL_REFERENCE_DRAFT_DIR = DATASETS_DIR / "Basketball Reference Draft"
TWENTY_YEARS_DRAFT_FILE = DATASETS_DIR / "20 Years of NBA Draft Data" / "draft-data-20-years.csv"
DRAFT_COMBINE_FILE = DATASETS_DIR / "NBA Draft Combine" / "Draft Combine - Kaggle.csv"
NBA_PLAYERS_FILE = DATASETS_DIR / "V2 NBA Player Database" / "NBA_PLAYERS.csv"
NBA_BOX_SCORE_DATASET_DIR = DATASETS_DIR / "NBA Dataset Box Scores and Stats (1947 - Today)"
NBA_BOX_SCORE_PLAYERS_FILE = NBA_BOX_SCORE_DATASET_DIR / "Players.csv"
NBA_BOX_SCORE_PLAYER_STATS_EXTENDED_FILE = NBA_BOX_SCORE_DATASET_DIR / "PlayerStatisticsExtended.csv"

# Milvus config. Defaults are local-friendly; docker-compose overrides host to
# the Milvus service name.
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "ncaa_nba_players")

# Model config
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.1"))
TRAIN_MAX_DRAFT_YEAR = int(os.getenv("TRAIN_MAX_DRAFT_YEAR", "2018"))
VALIDATION_START_DRAFT_YEAR = int(os.getenv("VALIDATION_START_DRAFT_YEAR", "2019"))
VALIDATION_END_DRAFT_YEAR = int(os.getenv("VALIDATION_END_DRAFT_YEAR", "2020"))
TEST_START_DRAFT_YEAR = int(os.getenv("TEST_START_DRAFT_YEAR", "2021"))
TEST_END_DRAFT_YEAR = int(os.getenv("TEST_END_DRAFT_YEAR", "2022"))

# Target definition thresholds
BPM_STAR_THRESHOLD = 4.0
MPG_STAR_THRESHOLD = 30.0
BPM_STARTER_THRESHOLD = 1.0
MPG_STARTER_THRESHOLD = 25.0
BPM_ROTATION_THRESHOLD = -2.0
MPG_ROTATION_THRESHOLD = 16.0
MINUTES_NON_NBA_THRESHOLD = 500.0
NBA_TARGET_ACTIVE_SEASONS = int(os.getenv("NBA_TARGET_ACTIVE_SEASONS", "3"))
NBA_TARGET_ACTIVE_SEASON_OFFSET = int(os.getenv("NBA_TARGET_ACTIVE_SEASON_OFFSET", "0"))

# SMOTE config
SMOTE_SAMPLING_STRATEGY = 0.7

# Vector embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Role labels in ordinal order. The fine labels are kept to build the target
# and compare historical experiments; the public final model uses grouped roles.
FINE_ROLE_LABELS = ["NON_NBA", "BENCH", "ROTATION", "STARTER_PLUS"]
GROUPED_ROLE_LABELS = ["NON_NBA", "NBA_ROLE_PLAYER", "STARTER_PLUS"]
ROLE_GROUP_MAP = {
    "NON_NBA": "NON_NBA",
    "BENCH": "NBA_ROLE_PLAYER",
    "ROTATION": "NBA_ROLE_PLAYER",
    "STARTER_PLUS": "STARTER_PLUS",
}
ROLE_LABELS = FINE_ROLE_LABELS

# Model feature contract shared by training, API, and processed-data builders.
MODEL_FEATURE_COLUMNS = [
    "pts_per40_z",
    "ast_per40_z",
    "reb_per40_z",
    "stl_per40_z",
    "blk_per40_z",
    "tov_per40_z",
    "ts_pct_z",
    "efg_pct_z",
    "usg_pct_z",
    "ast_tov_ratio_z",
    "stocks_per40_z",
    "three_par_z",
    "ftr_z",
    "rec_rank_z",
    "is_unranked_z",
    "official_games_z",
    "official_mpg_z",
    "official_start_rate_z",
    "height_in_z",
    "weight_lb_z",
    "wingspan_in_z",
    "bmi_z",
    "wingspan_diff_z",
    "age_at_draft_z",
    "team_adj_o_z",
    "team_adj_d_z",
    "team_tempo_z",
    "team_power_rating_z",
    "strength_of_schedule_z",
]

NUMERIC_VECTOR_DIM = len(MODEL_FEATURE_COLUMNS)
