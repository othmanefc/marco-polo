import unittest

from src.bm25 import BM25
from src.preprocesser import Corpus


class TestBM25(unittest.TestCase):

    def test_init_corpus(self):
        corpus = {
            1: "I love pudding",
            2: "I go to the beach",
            3: "the weather is nice today"
        }
        corp = Corpus(corpus)
        bm = BM25(corp)
        self.assertEqual(len(corpus), len(bm.len_docs))
        self.assertEqual(len(corpus), len(bm.doc_freqs))


if __name__ == "__main__":
    unittest.main()
