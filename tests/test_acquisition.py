import numpy as np

from petabite.active_learning import AcquisitionFactory


def test_random_acquisition_reproducible():
    cls = AcquisitionFactory("random")
    acq = cls(seed=7)
    scores_a = acq.score(pool_size=10)
    acq_b = cls(seed=7)
    scores_b = acq_b.score(pool_size=10)
    assert np.allclose(scores_a, scores_b)


def test_uncertainty_acquisition_ranks_by_variance():
    cls = AcquisitionFactory("uncertainty")
    acq = cls()
    variances = np.array([0.1, 0.9, 0.5])
    scores = acq.score_from_variance(variances)
    assert np.argmax(scores) == 1
