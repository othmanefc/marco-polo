import unittest

import numpy as np
from rank_bm25 import BM25L

from src.bm25 import BM25
from src.preprocesser import Corpus, Query
from src.templates import CORPUS, QUERY


class TestBM25(unittest.TestCase):

    corpus = CORPUS
    query = QUERY
    quer = Query(list(query.keys())[0], query[list(query.keys())[0]])
    corp = Corpus(corpus)
    bm = BM25(corp)
    bm_base = BM25L(corp.tokens.values())

    def test_init_corpus(self):
        self.assertEqual(len(self.corpus), len(self.bm.len_docs))
        self.assertEqual(len(self.corpus), len(self.bm.doc_freqs))

    def test_score(self):
        score = self.bm.score(self.quer)
        self.assertEqual(len(score), len(self.corpus))
        bml_score = self.bm_base.get_scores(self.quer.tokens)
        np.testing.assert_allclose(bml_score, score, rtol=2e-3)

    def test_top_n(self):
        N = 3
        top_score = self.bm.top_n(self.quer, n=N)
        bml_score = self.bm_base.get_scores(self.quer.tokens)
        bml_top_n = bml_score[::1][:N]
        np.testing.assert_allclose([score[1] for score in top_score],
                                   bml_top_n,
                                   rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
