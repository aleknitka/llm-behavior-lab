from __future__ import annotations

import hashlib
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator


class RandUniformRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["rand_uniform_range"] = "rand_uniform_range"
    left: StrictInt
    right: StrictInt

    def __init__(self, left: int, right: int, **data: object) -> None:
        super().__init__(left=left, right=right, **data)

    @model_validator(mode="after")
    def validate_bounds(self) -> RandUniformRange:
        if self.left > self.right:
            raise ValueError("left must be less than or equal to right")
        return self

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.left, self.right)


PersonaFieldValue = RandUniformRange | bool | int | str | None


def stable_random(seed: int, *parts: object) -> random.Random:
    key = ":".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(key.encode()).digest()
    return random.Random(int.from_bytes(digest))
