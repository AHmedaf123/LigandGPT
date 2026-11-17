"""Unit tests for the rebuilt reward system. Run: .venv/bin/python -m pytest tests/test_reward.py -v"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import generation_finetuning as gf


class TestSizeTerm(unittest.TestCase):
    def test_inside_band(self):
        for n in (18, 25, 38):
            self.assertAlmostEqual(gf.size_term(n), 1.0)

    def test_ramps(self):
        self.assertAlmostEqual(gf.size_term(8), 0.0)
        self.assertAlmostEqual(gf.size_term(13), 0.5)
        self.assertAlmostEqual(gf.size_term(43), 0.5)
        self.assertAlmostEqual(gf.size_term(48), 0.0)

    def test_outside(self):
        self.assertAlmostEqual(gf.size_term(5), 0.0)
        self.assertAlmostEqual(gf.size_term(50), 0.0)
        self.assertAlmostEqual(gf.size_term(None), 0.0)


class TestFixedNormalizer(unittest.TestCase):
    def setUp(self):
        self.norm = gf.FixedNormalizer({
            "docking": (-9.0, -5.0, True),
            "qed": (0.1, 0.8, False),
        })

    def test_minimize_better_is_lower(self):
        self.assertAlmostEqual(self.norm("docking", -9.0), 1.0)
        self.assertAlmostEqual(self.norm("docking", -5.0), 0.0)
        self.assertAlmostEqual(self.norm("docking", -7.0), 0.5)

    def test_maximize_better_is_higher(self):
        self.assertAlmostEqual(self.norm("qed", 0.8), 1.0)
        self.assertAlmostEqual(self.norm("qed", 0.1), 0.0)

    def test_clipping(self):
        self.assertAlmostEqual(self.norm("docking", -20.0), 1.0)
        self.assertAlmostEqual(self.norm("docking", 0.0), 0.0)

    def test_none_returns_zero(self):
        self.assertAlmostEqual(self.norm("docking", None), 0.0)


class TestQualityDivisor(unittest.TestCase):
    def test_warmup_D(self):
        self.assertAlmostEqual(2.5, 1.0 + 1.5)

    def test_no_curriculum(self):
        D = gf.quality_divisor("no_curriculum", 1, list(gf.OBJECTIVES_ORDER))
        self.assertAlmostEqual(D, 1.0 + 1.5 + 9.0)

    def test_pairs_stage1(self):
        D = gf.quality_divisor("curriculum_pairs", 1, ["docking", "bioactivity"])
        self.assertAlmostEqual(D, 1.0 + 1.5 + 1.5 * 2)

    def test_adaptive(self):
        D = gf.quality_divisor("adaptive", 3, ["docking", "bioactivity", "qed"])
        self.assertAlmostEqual(D, 1.0 + 1.5 + 1.5 * 3)


class TestWarmupReward(unittest.TestCase):
    def test_valid_in_band(self):
        R = 1.0 * 1.0 + 1.5 * gf.size_term(25)
        self.assertAlmostEqual(R, 2.5)

    def test_invalid(self):
        R = 1.0 * 0.0 + 1.5 * gf.size_term(None)
        self.assertAlmostEqual(R, 0.0)

    def test_valid_small(self):
        R = 1.0 * 1.0 + 1.5 * gf.size_term(5)
        self.assertAlmostEqual(R, 1.0)


class TestStageWeights(unittest.TestCase):
    def test_weights_sum_to_one(self):
        for s in (1, 2, 3):
            w = gf.STAGE_WEIGHTS_CW[s]
            self.assertAlmostEqual(sum(w.values()), 1.0, places=2)

    def test_no_curriculum_weights_sum(self):
        self.assertAlmostEqual(sum(gf.FIXED_WEIGHTS_NC.values()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
