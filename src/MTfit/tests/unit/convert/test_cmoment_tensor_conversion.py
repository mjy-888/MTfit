"""
test_cmoment_tensor_conversion.py
*****************

Tests for src/convert/cmoment_tensor_conversion.pyx

These tests exercise the public API of the Cython extension module.
The test logic was previously embedded in the .pyx file and has been
moved here for Cython 3.x compatibility (cdef statements are not
allowed inside Python classes in Cython 3).
"""

import unittest
import math

import numpy as np

try:
    from MTfit.convert.cmoment_tensor_conversion import (
        Tape_MT6,
        MT6_TNPE,
        E_GD,
        TP_SDR,
        SDR_SDR,
        MT6c_D6,
        isotropic_c,
        is_isotropic_c,
        c_norm,
        MT_output_convert,
    )
    _HAS_C_MODULE = True
except ImportError:
    _HAS_C_MODULE = False


@unittest.skipUnless(_HAS_C_MODULE, "No cmoment_tensor_conversion module")
class cMomentTensorConvertTestCase(unittest.TestCase):
    """Tests ported from the embedded Cython test class."""

    def test_cTape_MT6(self):
        # Single-element arrays to match the function signature
        gamma = np.array([0.12])
        delta = np.array([0.43])
        kappa = np.array([0.76])
        h = np.array([0.63])
        sigma = np.array([0.75])
        mt = Tape_MT6(gamma, delta, kappa, h, sigma)
        # mt is (6, 1) -- squeeze to 1-d
        m = mt[:, 0]
        self.assertAlmostEqual(m[0], -0.3637, 4)
        self.assertAlmostEqual(m[1], 0.4209, 4)
        self.assertAlmostEqual(m[2], 0.6649, 4)
        self.assertAlmostEqual(m[3], 0.3533, 4)
        self.assertAlmostEqual(m[4], -0.1952, 4)
        self.assertAlmostEqual(m[5], -0.2924, 4)

    def test_c_MT6_TNPE(self):
        MT = np.array([[1., 0., -1., 0., 0., 0.],
                       [0, 2.0, -1.0, 0., 1.0, 0.]]).T
        T, N, P, E = MT6_TNPE(MT)
        self.assertAlmostEqual(E[0, 0], 1)
        self.assertAlmostEqual(E[1, 0], 0)
        self.assertAlmostEqual(E[2, 0], -1)
        self.assertAlmostEqual(np.abs(T[0, 0]), 1)
        self.assertAlmostEqual(np.abs(T[1, 0]), 0)
        self.assertAlmostEqual(np.abs(T[2, 0]), 0)
        self.assertAlmostEqual(np.abs(N[0, 0]), 0)
        self.assertAlmostEqual(np.abs(N[1, 0]), 1)
        self.assertAlmostEqual(np.abs(N[2, 0]), 0)
        self.assertAlmostEqual(np.abs(P[0, 0]), 0)
        self.assertAlmostEqual(np.abs(P[1, 0]), 0)
        self.assertAlmostEqual(np.abs(P[2, 0]), 1)
        # Second Event
        self.assertAlmostEqual(E[0, 1], 2)
        self.assertAlmostEqual(E[1, 1], 0.366025403784439)
        self.assertAlmostEqual(E[2, 1], -1.36602540378444)
        self.assertAlmostEqual(np.abs(T[0, 1]), 0)
        self.assertAlmostEqual(np.abs(T[1, 1]), 1)
        self.assertAlmostEqual(np.abs(T[2, 1]), 0)
        self.assertAlmostEqual(np.abs(N[0, 1]), 0.888073833977115)
        self.assertAlmostEqual(np.abs(N[1, 1]), 0)
        self.assertAlmostEqual(np.abs(N[2, 1]), 0.459700843380983)
        self.assertAlmostEqual(np.abs(P[0, 1]), 0.459700843380983)
        self.assertAlmostEqual(np.abs(P[1, 1]), 0)
        self.assertAlmostEqual(np.abs(P[2, 1]), 0.888073833977115)

    def test_c_E_GD(self):
        E = np.array([[1., 0., -1.], [1., 1., 1.],
                       [1., -1., -1.], [-1., -1., -1.]]).T
        g, d = E_GD(E)
        self.assertAlmostEqual(float(g[0]), 0.)
        self.assertAlmostEqual(float(d[0]), 0.)
        self.assertAlmostEqual(float(g[1]), 0.)
        self.assertAlmostEqual(float(d[1]), math.pi / 2)
        self.assertAlmostEqual(float(g[2]), -0.523598775598299)
        self.assertAlmostEqual(float(d[2]), -0.339836909454122)
        self.assertAlmostEqual(float(g[3]), 0.)
        self.assertAlmostEqual(float(d[3]), -math.pi / 2)

    def test_c_TP_SDR(self):
        T = np.array([[0.235702260395516, 0.],
                      [0.235702260395516, 1.0],
                      [0.942809041582063, 0.]])
        P = np.array([[0.707106781186547, 1.],
                      [-0.707106781186547, 0],
                      [0., 0.]])
        s, d, r = TP_SDR(T, P)
        self.assertAlmostEqual(float(s[0]), 1.1071487177940911)
        self.assertAlmostEqual(float(d[0]), 0.84106867056793)
        self.assertAlmostEqual(float(r[0]), 2.0344439357957032)
        self.assertAlmostEqual(float(s[1]), 5.497787143782138)
        self.assertAlmostEqual(float(d[1]), math.pi / 2)
        self.assertAlmostEqual(float(r[1]), -math.pi)

    def test_c_SDR_SDR(self):
        s2, d2, r2 = SDR_SDR(
            np.array([206.565051177078 * math.pi / 180]),
            np.array([48.1896851042214 * math.pi / 180]),
            np.array([63.434948822922 * math.pi / 180]))
        self.assertAlmostEqual(float(s2), 63.434948822922 * math.pi / 180)
        self.assertAlmostEqual(float(d2), 48.1896851042214 * math.pi / 180)
        self.assertAlmostEqual(float(r2), 116.565051177078 * math.pi / 180)
        s2, d2, r2 = SDR_SDR(
            np.array([63.434948822922 * math.pi / 180]),
            np.array([48.1896851042214 * math.pi / 180]),
            np.array([116.565051177078 * math.pi / 180]))
        self.assertAlmostEqual(float(s2), 206.565051177078 * math.pi / 180)
        self.assertAlmostEqual(float(d2), 48.1896851042214 * math.pi / 180)
        self.assertAlmostEqual(float(r2), 63.434948822922 * math.pi / 180)

    def test_MT_output_convert(self):
        MT = np.array([[1., 0., -1., 0., 0., 0.]]).T
        result = MT_output_convert(MT)
        self.assertIn('g', result)
        self.assertIn('d', result)
        self.assertIn('k', result)
        self.assertIn('h', result)
        self.assertIn('s', result)
        self.assertIn('u', result)
        self.assertIn('v', result)
        self.assertEqual(len(result['g']), 1)

    def test_isotropic_c(self):
        c = isotropic_c(l=1., mu=1.)
        self.assertEqual(len(c), 21)
        self.assertTrue(is_isotropic_c(c))
        self.assertGreater(c_norm(c), 0)


def test_suite(verbosity=2):
    suite = [unittest.TestLoader().loadTestsFromTestCase(cMomentTensorConvertTestCase)]
    suite = unittest.TestSuite(suite)
    return suite
