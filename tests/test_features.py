import numpy as np

from vehicle_behavior.features import (
    calculate_lateral_speeds,
    interpolate_missing_values,
    parse_coordinate_string,
)


def test_interpolate_missing_values():
    sequence = [1.0, None, 3.0, np.nan, 5.0]
    result = interpolate_missing_values(sequence)
    assert len(result) == 5
    assert result[0] == 1.0
    assert result[2] == 3.0
    assert result[4] == 5.0
    assert result[1] != 0.0
    assert result[3] != 0.0


def test_parse_coordinate_string():
    coords = "[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]"
    parsed = parse_coordinate_string(coords)
    assert len(parsed) == 3
    assert parsed[0] == (1.0, 2.0)


def test_calculate_lateral_speeds():
    coords = [(0.0, 0.0), (3.0, 4.0), (3.0, 8.0)]
    speeds = calculate_lateral_speeds(coords)
    assert len(speeds) == 2
    assert speeds[0] == 5.0
