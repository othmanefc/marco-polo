from typing import Dict, List, Sequence, Tuple

import math
import numpy as np
from statistics import mean
from rank_bm25 import BM25L

from src.preprocesser import Corpus, Query


class BM25(object):

    def __init__(self,
                 corpus: Corpus,
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

        self.corpus = corpus
        self.init_corpus(self.corpus)
        self.avg_len = mean(self.len_docs)
        self.compute_idf()

    def init_corpus(self, corpus: Corpus) -> None:
        for i, (pid, doc) in enumerate(corpus.tokens.items()):
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
            score += self.idf.get(term, 0) * term_freq * (
                (self.k1 + 1) * (c + self.delta) / (self.k1 + c + self.delta))
        return score

    def top_n(self, query: Query, n: int = None) -> List[Tuple[int, float]]:
        if n is None:
            n = len(self.corpus)
        scores = self.score(query)
        top = np.argsort(scores)[::-1][:n]
        return [(self.corpus.pids[i], scores[i]) for i in top]

    def compute_idf(self) -> None:
        for word, freq in self.word_idx.items():
            idf = math.log(len(self.corpus) + 1) - math.log(freq + 0.5)
            self.idf[word] = idf


if __name__ == "__main__":
    corpus = {
        1: "I love pudding",
        2: "I go to the beach",
        3: "the weather is nice today"
    }
    query = {3000: "What is love ?"}
    quer = Query(list(query.keys())[0], query[list(query.keys())[0]])
    corp = Corpus(corpus)
    bm25 = BM25(corp)
    score = bm25.score(quer)
    print("score", score)
    print("top score", bm25.top_n(quer, n=3))
    bm25l = BM25L(corp.tokens.values())
    score_pip = bm25l.get_scores(quer.tokens)
    print("score_pip", score_pip)
