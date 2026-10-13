"""API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlayerStats(BaseModel):
    """Input profile used to predict an NBA role from NCAA indicators."""

    player_name: str = Field(..., min_length=1, examples=["Test Player"])
    position: str = Field(..., min_length=1, max_length=10, examples=["G"])
    college: str = Field("Unknown", examples=["Villanova"])
    season: int = Field(2026, ge=2000, le=2035)
    draft_year: int = Field(2026, ge=2000, le=2035)
    draft_pick: int = Field(0, ge=0, le=60, description="0 when the player has not been drafted yet")
    age: float = Field(21.0, ge=16.0, le=30.0)
    age_at_draft: float | None = Field(None, ge=16.0, le=30.0)
    rec_rank: float | None = Field(None, ge=0.0, le=100.0)
    is_unranked: float | None = Field(None, ge=0.0, le=1.0)
    official_games: float | None = Field(None, ge=0.0)
    official_mpg: float | None = Field(None, ge=0.0)
    official_start_rate: float | None = Field(None, ge=0.0, le=1.0)

    height_in: float = Field(..., gt=0.0, examples=[75.0])
    weight_lb: float = Field(..., gt=0.0, examples=[190.0])
    wingspan_in: float | None = Field(None, gt=0.0, examples=[78.0])

    pts_per40: float = Field(..., ge=0.0, examples=[20.5])
    ast_per40: float = Field(..., ge=0.0, examples=[4.8])
    reb_per40: float = Field(..., ge=0.0, examples=[5.2])
    stl_per40: float = Field(1.2, ge=0.0)
    blk_per40: float = Field(0.4, ge=0.0)
    tov_per40: float = Field(2.5, ge=0.0)

    ts_pct: float = Field(..., ge=0.0, examples=[58.0])
    efg_pct: float = Field(54.0, ge=0.0)
    usg_pct: float = Field(..., ge=0.0, examples=[24.0])
    ast_tov_ratio: float | None = Field(None, ge=0.0)
    three_par: float = Field(0.35, ge=0.0)
    ftr: float = Field(32.0, ge=0.0)

    team_adj_o: float = Field(..., examples=[115.0])
    team_adj_d: float = Field(96.0)
    team_tempo: float = Field(68.0)
    team_power_rating: float = Field(0.80)
    strength_of_schedule: float = Field(0.0)


class PredictionResponse(BaseModel):
    """Prediction returned by the role model."""

    player_name: str
    predicted_role: str
    probabilities: dict[str, float]
    confidence: float
    model_name: str
    feature_source: str


class ComparablePlayer(BaseModel):
    """Comparable player returned from Milvus."""

    player_id: str | None = None
    player_name: str | None = None
    position: str | None = None
    target_role: str | None = None
    draft_year: int | None = None
    college: str | None = None
    score: float


class ComparablePlayersResponse(BaseModel):
    """Comparable-player search response."""

    player_name: str
    matched_player_id: str | None = None
    top_k: int
    comparables: list[ComparablePlayer]


class HealthResponse(BaseModel):
    """Application health response."""

    status: str
    model_loaded: bool
    model_name: str | None = None
    milvus_collection: str
