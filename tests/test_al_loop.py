from pathlib import Path

import numpy as np

from petabite.active_learning import ActiveLearningLoop


def _records(n):
    return [{"id": str(i), "sequence": "ACDEFG"} for i in range(n)]


def test_al_loop_run_round_exports_query(tmp_path: Path):
    loop = ActiveLearningLoop(
        acquisition_name="uncertainty", query_size=2, output_dir=tmp_path
    )
    pool = _records(5)
    variances = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    selected, query_path = loop.run_round(
        labeled=_records(2), pool=pool, round_idx=1, variances=variances
    )
    assert selected == [1, 3]
    assert query_path.exists()
    assert query_path.name == "query_round_1.csv"
