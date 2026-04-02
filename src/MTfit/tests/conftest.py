"""
Shared pytest fixtures for MTfit test suite.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_mt6_dc():
    """A known double-couple moment tensor 6-vector (normalized).

    Represents a vertical strike-slip fault:
    strike=0, dip=90, rake=0
    """
    mt = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])
    return mt / np.linalg.norm(mt)


@pytest.fixture
def sample_mt33_dc():
    """A known double-couple 3x3 moment tensor.

    Corresponding to sample_mt6_dc.
    """
    s2 = 1.0 / np.sqrt(2.0)
    return np.array([
        [0.0, s2, 0.0],
        [s2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])


@pytest.fixture
def sample_mt6_clvd():
    """A known CLVD moment tensor 6-vector (normalized)."""
    mt = np.array([[2.0], [-1.0], [-1.0], [0.0], [0.0], [0.0]])
    return mt / np.linalg.norm(mt)


@pytest.fixture
def sample_mt6_isotropic():
    """A known isotropic (explosive) moment tensor 6-vector (normalized)."""
    mt = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
    return mt / np.linalg.norm(mt)


@pytest.fixture
def sample_polarity_data():
    """Standard polarity data dictionary using ndarray.

    Contains 4 stations with P-polarity observations.
    """
    return {
        'PPolarity': {
            'Measured': np.array([[1.0], [-1.0], [1.0], [-1.0]]),
            'Error': np.array([[0.05], [0.05], [0.1], [0.1]]),
            'Stations': {
                'Name': ['S1', 'S2', 'S3', 'S4'],
                'Azimuth': np.array([[45.0], [135.0], [225.0], [315.0]]),
                'TakeOffAngle': np.array([[30.0], [30.0], [30.0], [30.0]]),
            },
        },
        'UID': 'test_event',
    }
