"""Request/response schemas for the wildlife camera evaluation API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The contract requires signed 16-bit integer feature values.
FeatureValue = Annotated[int, Field(ge=-32768, le=32767, strict=True)]


class PhotoIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_id: str = Field(min_length=1, max_length=128)
    species: str = Field(min_length=1, max_length=128)
    features: list[FeatureValue] = Field(min_length=32, max_length=32)

    @field_validator("photo_id", "species")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class PairIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_a: str = Field(min_length=1, max_length=128)
    photo_b: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _different_photos(self) -> "PairIn":
        if self.photo_a == self.photo_b:
            raise ValueError("a photo pair must contain two different photo IDs")
        return self


class BatchCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Client-supplied submission identifier; generated when omitted.
    batch_id: str | None = Field(default=None, min_length=1, max_length=128)
    photos: list[PhotoIn] = Field(min_length=1, max_length=1000)
    pairs: list[PairIn] = Field(default_factory=list, max_length=499_500)

    @model_validator(mode="after")
    def _validate_batch(self) -> "BatchCreateRequest":
        ids = [photo.photo_id for photo in self.photos]
        if len(set(ids)) != len(ids):
            raise ValueError("photo_id values must be unique within a batch")

        known = set(ids)
        seen_pairs: set[tuple[str, str]] = set()
        for pair in self.pairs:
            if pair.photo_a not in known or pair.photo_b not in known:
                raise ValueError(
                    "every pair endpoint must refer to a photo in this batch: "
                    f"{pair.photo_a!r}, {pair.photo_b!r}"
                )
            normalized = (
                (pair.photo_a, pair.photo_b)
                if pair.photo_a < pair.photo_b
                else (pair.photo_b, pair.photo_a)
            )
            if normalized in seen_pairs:
                raise ValueError(f"duplicate photo pair: {normalized[0]!r}, {normalized[1]!r}")
            seen_pairs.add(normalized)
        return self


class CorrectionRequest(BaseModel):
    """One reviewer correction against an explicitly expected version."""

    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1, strict=True)
    action: Literal["add", "remove"]
    photo_a: str = Field(min_length=1, max_length=128)
    photo_b: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_correction(self) -> "CorrectionRequest":
        if self.photo_a == self.photo_b:
            raise ValueError("a correction must refer to two different photo IDs")
        return self


class PredictionOut(BaseModel):
    photo_id: str
    known_species: str
    predicted_species: str
    nearest_photo_id: str
    distance: float
    correct: bool


class GroupOut(BaseModel):
    photo_ids: list[str]
    size: int
    correct: int | None
    accuracy: float | None


class EvaluationOut(BaseModel):
    batch_id: str
    version: int
    photo_count: int
    group_count: int
    evaluable: bool
    reason: str | None = None
    message: str | None = None
    overall_accuracy: float | None = None
    groups: list[GroupOut]
    predictions: list[PredictionOut]
    created_at: str
    version_created_at: str


class CorrectionOut(BaseModel):
    batch_id: str
    previous_version: int
    version: int
    action: Literal["add", "remove"]
    photo_a: str
    photo_b: str
    recomputed_photo_ids: list[str]
    photo_count: int
    group_count: int
    evaluable: bool
    reason: str | None = None
    message: str | None = None
    overall_accuracy: float | None = None
    version_created_at: str
