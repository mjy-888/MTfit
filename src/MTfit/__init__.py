"""
**Restricted:  For Non-Commercial Use Only**
This code is protected intellectual property and is available solely for teaching
and non-commercially funded academic research purposes.

Applications for commercial use should be made to Schlumberger or the University of Cambridge.
"""
import sys
import multiprocessing
import json

from ._version import __version__
from .run import MTfit  # noqa: F401

__all__ = ['algorithms', 'convert', 'extensions', 'plot', 'utilities', 'inversion', 'probability', 'sampling']


def get_details() -> dict:
    """Return a dictionary of MTfit installation details."""
    c_extensions = []
    for name, module_path in [
        ('cmarkov_chain_monte_carlo', 'MTfit.algorithms.cmarkov_chain_monte_carlo'),
        ('cprobability', 'MTfit.probability.cprobability'),
        ('cmoment_tensor_conversion', 'MTfit.convert.cmoment_tensor_conversion'),
        ('cscatangle', 'MTfit.extensions.cscatangle'),
    ]:
        try:
            __import__(module_path)
            c_extensions.append(name)
        except Exception:
            pass

    dependency_versions = {}
    import numpy as np
    dependency_versions['numpy'] = np.__version__
    import scipy
    dependency_versions['scipy'] = scipy.__version__

    for pkg_name in ('matplotlib', 'cython', 'pyqsub', 'sphinx', 'h5py', 'hdf5storage'):
        try:
            mod = __import__(pkg_name)
            dependency_versions[pkg_name] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            pass

    details = {
        'version': __version__,
        'c_extensions present': c_extensions,
        'platform': sys.platform,
        'num_threads': multiprocessing.cpu_count(),
        'python version': sys.version,
        'python version info': sys.version_info,
        'dependency info': dependency_versions,
    }
    if sys.platform.startswith('win'):
        details['windows version'] = sys.getwindowsversion()
    return details


def get_details_json() -> str:
    return json.dumps(get_details(), indent=4)
