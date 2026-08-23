"""Single config source of truth, validated with Pydantic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Mode = Literal["auto", "entailment", "consistency"]


class Config(BaseModel):
    """HaloGuard configuration.

    Verdict mapping: risk < threshold -> PASS,
    threshold <= risk < block_threshold -> FLAG,
    risk >= block_threshold -> BLOCK.
    """

    threshold: float = Field(default=0.7, description="PASS/FLAG risk boundary")
    block_threshold: float = Field(default=0.9, description="FLAG/BLOCK risk boundary")
    mode: Mode = Field(default="auto", description="auto | entailment | consistency")
    timeout_s: float = Field(default=5.0, gt=0, description="Inference timeout in seconds")
    strict_mode: bool = Field(default=False, description="Fail closed instead of returning UNKNOWN")
    max_input_chars: int = Field(default=100_000, gt=0, description="Input size cap in characters")

    @model_validator(mode="after")
    def _check_thresholds(self) -> Config:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if not self.threshold < self.block_threshold <= 1.0:
            raise ValueError("block_threshold must be in (threshold, 1]")
        return self
