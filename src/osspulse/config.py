"""Load and validate configs/cohort.yaml and configs/packages.yaml."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

CONFIG_PATH = Path("configs/cohort.yaml")
PACKAGES_PATH = Path("configs/packages.yaml")


class CohortConfig(BaseModel):
    language: str
    created_from: date
    created_to: date
    search_shard_by: Literal["month", "year"] = "month"
    search_min_stars: int = Field(ge=0)
    search_max_stars_escalation: int = Field(ge=0)
    inclusion_min_stars_90d: int = Field(ge=0)
    inclusion_min_contributors_90d: int = Field(ge=1)
    inactivity_days: int = Field(ge=1)
    sample_size: int = Field(ge=1)
    sample_seed: int = Field(ge=0)
    observation_end: date

    @model_validator(mode="after")
    def _check_dates(self) -> CohortConfig:
        if self.created_from >= self.created_to:
            raise ValueError("created_from must be < created_to")
        if self.observation_end <= self.created_from:
            raise ValueError("observation_end must be after created_from")
        if self.search_max_stars_escalation <= self.search_min_stars:
            raise ValueError("escalation threshold must be > search_min_stars")
        return self


class DownloadConfig(BaseModel):
    source: Literal["gharchive_hourly"] = "gharchive_hourly"
    url_template: str = "https://data.gharchive.org/{date}-{hour}.json.gz"
    keep_raw: bool = False
    manifest_path: Path = Path("data/raw/gharchive/_download_manifest.jsonl")
    max_parallel: int = Field(default=2, ge=1, le=8)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    request_timeout_sec: float = Field(default=60.0, gt=0)

    @field_validator("url_template")
    @classmethod
    def _url_placeholders(cls, v: str) -> str:
        if "{date}" not in v or "{hour}" not in v:
            raise ValueError("url_template must contain {date} and {hour}")
        return v


class SearchApiConfig(BaseModel):
    per_page: int = Field(ge=1, le=100)
    max_pages: int = Field(ge=1, le=10)
    min_interval_sec: float = Field(gt=0)


class SourcesConfig(BaseModel):
    pypi: Literal["pypistats"] = "pypistats"
    ci_detection: Literal["gharchive_push"] = "gharchive_push"


class BotsConfig(BaseModel):
    known_logins: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class PackagesConfig(BaseModel):
    established: list[str] = Field(default_factory=list)
    newer: list[str] = Field(default_factory=list)

    def all_names(self) -> list[str]:
        return list(dict.fromkeys(self.established + self.newer))


class Config(BaseModel):
    cohort: CohortConfig
    download: DownloadConfig
    search_api: SearchApiConfig
    sources: SourcesConfig
    bots: BotsConfig = BotsConfig()
    logging: LoggingConfig

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> Config:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path} did not parse to a mapping")
        return cls.model_validate(raw)

    @classmethod
    def load_packages(cls, path: Path | str = PACKAGES_PATH) -> PackagesConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path} did not parse to a mapping")
        return PackagesConfig.model_validate(raw)
