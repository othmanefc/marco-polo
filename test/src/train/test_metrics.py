import unittest

from src.train.metrics import mrr


class TestMRR(unittest.TestCase):

    def test_mmr(self):
        true = ["2", "4", "6", "1"]
        pred = ["9", "4", "1", "3"]
        res = mrr(true, pred)
        self.assertEqual(res, 1/2)
