import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.get_observed_stats import aggregate_sites_to_tip_results


def test_aggregate_sites_to_tip_results_builds_tip_style_vectors():
    matrix = pd.DataFrame(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]],
        index=["strain_a", "strain_b", "strain_c"],
        columns=["site1", "site2", "site3", "site4"],
    )
    matrix_meta = pd.DataFrame(
        {
            "site_id": ["site1", "site2", "site3", "site4"],
            "context": ["region_1", "region_1", "region_2", "region_2"],
            "side": [5, 5, 5, 5],
        }
    )

    result = aggregate_sites_to_tip_results(matrix, matrix_meta, tip_names=["strain_a", "strain_b", "strain_c"], side=5)

    expected = {
        "strain_a": np.array([1, 1], dtype=int),
        "strain_b": np.array([1, 1], dtype=int),
        "strain_c": np.array([1, 0], dtype=int),
    }

    assert set(result) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(result[name], expected[name])
