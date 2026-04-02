"""unittest_utils
******************
Provides test functions for running and debugging unit tests.
"""

# **Restricted:  For Non-Commercial Use Only**
# This code is protected intellectual property and is available solely for teaching
# and non-commercially funded academic research purposes.
#
# Applications for commercial use should be made to Schlumberger or the University of Cambridge.

import unittest
import importlib
import traceback

import numpy as np


def get_extension_skip_if_args(module: str) -> tuple[bool, str]:
    reason = 'No C extension available'
    try:
        c_extension = importlib.import_module(module)
    except ImportError:
        c_extension = False
    except Exception:
        reason += f'\n=======\nException loading C extension = \n{traceback.format_exc()}\n=======\n'
        c_extension = False
    return (not c_extension, reason)


class TestCase(unittest.TestCase):

    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):

        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            if isinstance(second, (list, float, int)):
                second = np.array(second)
            if isinstance(first, (list, float, int)):
                first = np.array(first)
            if len([u for u in second.shape if u != 1]) == len([u for u in first.shape if u != 1]):
                if places is not None:
                    np.testing.assert_array_almost_equal(np.array(first).squeeze(), np.array(second).squeeze(), places)
                else:
                    np.testing.assert_array_almost_equal(np.array(first).squeeze(), np.array(second).squeeze())
            else:
                if places is not None:
                    np.testing.assert_array_almost_equal(first, second, places)
                else:
                    np.testing.assert_array_almost_equal(first, second)
            return
        elif isinstance(first, dict) and isinstance(second, dict):
            self.assertEqual(sorted(first.keys()), sorted(second.keys()), 'Dictionary keys do not match')
            for key in first.keys():
                self.assertAlmostEqual(first[key], second[key])
            return
        elif isinstance(first, list) and isinstance(second, list):
            super().assertAlmostEqual(set(first), set(second), msg, delta)
        else:
            super().assertAlmostEqual(first, second, places, msg, delta)

    def assertEqual(self, first, second, msg=None):
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            np.testing.assert_array_equal(first, second)
            return
        elif isinstance(first, dict) and isinstance(second, dict):
            self.assertEqual(
                sorted(first.keys()), sorted(second.keys()), 'Dictionary keys do not match')
            for key in first.keys():
                self.assertEqual(first[key], second[key])
            return
        elif isinstance(first, list) and isinstance(second, list):
            super().assertEqual(first, second, msg)
        else:
            super().assertAlmostEqual(first, second, msg)

    def assertAlmostEquals(self, *args, **kwargs):
        return self.assertAlmostEqual(*args, **kwargs)

    def assertEquals(self, *args, **kwargs):
        return self.assertEqual(*args, **kwargs)

    def assertVectorEquals(self, first, second, *args):
        first = np.asarray(first)
        second = np.asarray(second)
        try:
            first_norm = np.sqrt(np.sum(np.multiply(first, first)))
            second_norm = np.sqrt(np.sum(np.multiply(second, second)))
            return self.assertAlmostEqual(first / first_norm, second / second_norm, *args)
        except AssertionError as e1:
            try:
                return self.assertAlmostEqual(-first / first_norm, second / second_norm, *args)
            except AssertionError as e2:
                raise AssertionError(f'{e1.args} or {e2.args}')
