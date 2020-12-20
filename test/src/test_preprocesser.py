import unittest

import numpy as np
from nltk.corpus import stopwords
import tensorflow_hub as hub

from src.preprocesser import Preprocesser, Query, Corpus
from src.templates import TEXT, QUERY, QUERY_TOKENS, CORPUS, CORPUS_TOKENS
from datasets import tokenization


class TestPreprocesser(unittest.TestCase):
    stopwords = set(stopwords.words("english"))

    def test_remove_punctuation(self):
        pr = Preprocesser()
        no_punc_output = pr.remove_punctuation(TEXT.get("punc"))
        self.assertEqual(no_punc_output, TEXT.get("no_punc"))

    def test_lemmatize(self):
        pr = Preprocesser()
        lem_output = pr.lemmatize(TEXT.get("no_punc").split())
        self.assertEqual(lem_output, TEXT.get("lemma"))

    def test_tokenize(self):
        pr = Preprocesser()
        wt_output = pr.tokenize(TEXT.get("no_punc"))
        self.assertEqual(wt_output, TEXT.get("wt"))

    def test_convert_to_bert_input(self):
        bert_layer = hub.KerasLayer(
            'https://tfhub.dev/tensorflow/small_bert/'
            'bert_en_uncased_L-2_H-128_A-2/1',
            trainable=True)
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

        real_bert = TEXT.get("bert_full")
        tokenized_op = Preprocesser.convert_to_bert_input(TEXT.get("no_punc"),
                                                          max_length=30,
                                                          tokenizer=tokenizer,
                                                          cls=True)
        self.assertEqual(tokenized_op, real_bert)

        tokenized_sub_op = Preprocesser.convert_to_bert_input(
            TEXT.get("no_punc"), max_length=3, tokenizer=tokenizer, cls=True)
        real_trunc = real_bert[:2] + real_bert[-1:]
        self.assertEqual(tokenized_sub_op, real_trunc)
        self.assertEqual(len(tokenized_sub_op), 3)


class TestQuery(unittest.TestCase):
    query = Query(list(QUERY.keys())[0], list(QUERY.values())[0])

    def test_preprocess(self):
        self.assertEqual(self.query.tokens,
                         QUERY_TOKENS.get(list(QUERY.keys())[0]))

    def test_update_answers(self):
        query = Query(list(QUERY.keys())[0], list(QUERY.values())[0])
        answers = [(i, np.random.normal(scale=3))
                   for i in np.random.randint(1000, size=20)]

        query.update_answers(answers,)
        self.assertEqual(query.answers, sorted(answers, key=lambda x: x[1]))

        new_answers = [(i, np.random.normal(scale=3))
                       for i in np.random.randint(1000, size=40)]
        query.update_answers(new_answers, n=10)
        self.assertEqual(
            query.answers,
            sorted(answers + new_answers, key=lambda x: x[1])[:10])


class TestCorpus(unittest.TestCase):

    def test_preprocess(self):
        corpus = Corpus(CORPUS)
        self.assertEqual(corpus.tokens, CORPUS_TOKENS)


if __name__ == '__main__':
    unittest.main()
