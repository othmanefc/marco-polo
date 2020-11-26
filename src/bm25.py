from typing import Dict, List, Sequence

import math
import numpy as np
from statistics import mean
from rank_bm25 import BM25L

from preprocesser import Corpus, Query


class BM25(object):
    def __init__(self,
                 corpus: List[str],
                 k1: float = 1.5,
                 b: float = 0.75,
                 delta: float = 0.5) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta

        self.doc_freqs: List[Dict[str, int]] = []
        self.word_idx: Dict[str, int] = {}
        self.len_docs: List[int] = []
        self.idf: Dict[str, float] = {}

        self.corpus = Corpus(corpus)
        self.init_corpus(self.corpus)
        self.avg_len = mean(self.len_docs)
        self.compute_idf()

    def init_corpus(self, corpus: Corpus) -> None:
        for i, doc in enumerate(corpus.tokens):
            freq: Dict[str, int] = {}
            for word in doc:
                freq_word_doc = freq.get(word, 0)
                freq[word] = freq_word_doc + 1
            for word in freq.keys():
                freq_word_corpus = self.word_idx.get(word, 0)
                self.word_idx[word] = freq_word_corpus + 1
            self.doc_freqs.append(freq)
            self.len_docs.append(len(doc))

    def score(self, query: Query) -> Sequence[float]:
        score = np.zeros(len(self.corpus))
        len_docs = np.array(self.len_docs)
        for term in query.tokens:
            term_freq = np.array(
                [doc_freq.get(term, 0) for doc_freq in self.doc_freqs])
            c = term_freq / (1 - self.b + self.b * (len_docs / self.avg_len))
            score += self.idf.get(term, 0) * ((self.k1 + 1) *
                                              (c + self.delta) /
                                              (self.k1 + c + self.delta))
        return score

    def compute_idf(self) -> None:
        for word, freq in self.word_idx.items():
            idf = math.log(len(self.corpus) + 1) - math.log(freq + 0.5)
            self.idf[word] = idf


if __name__ == "__main__":
    corpus = [
        "I love pudding", "I go to the beach", "the weather is nice today"
    ]
    query = "What is love ?"
    quer = Query(query)
    corp = Corpus(corpus)
    bm25 = BM25(corpus)
    score = bm25.score(quer)
    print("score", score)
    bm25l = BM25L(corp.tokens)
    score_pip = bm25l.get_scores(quer.tokens)
    print("score_pip", score_pip)