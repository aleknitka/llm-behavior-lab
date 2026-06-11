import random
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from llm_behavior_lab.personas.value_specs import RandUniformRange


def test_rand_uniform_range_round_trips_through_pydantic_json() -> None:
    adapter = TypeAdapter(RandUniformRange)
    value = RandUniformRange(20, 25)

    serialized = adapter.dump_json(value)

    assert serialized == b'{"type":"rand_uniform_range","left":20,"right":25}'
    assert adapter.validate_python(
        {"type": "rand_uniform_range", "left": 20, "right": 25}
    ) == value
    assert adapter.validate_json(serialized) == value


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (20.0, 25),
        (20, 25.0),
        (True, 25),
        (20, False),
        (25, 20),
    ],
)
def test_rand_uniform_range_rejects_invalid_bounds(left: object, right: object) -> None:
    with pytest.raises(ValidationError):
        RandUniformRange(cast(Any, left), cast(Any, right))


def test_rand_uniform_range_equal_bounds_are_constant() -> None:
    value = RandUniformRange(7, 7)

    assert value.sample(random.Random(1)) == 7
    assert value.sample(random.Random(999)) == 7


def test_rand_uniform_range_sampling_is_inclusive() -> None:
    value = RandUniformRange(20, 25)
    samples = {value.sample(random.Random(seed)) for seed in range(100)}

    assert min(samples) == 20
    assert max(samples) == 25
    assert samples <= set(range(20, 26))
