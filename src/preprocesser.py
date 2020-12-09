from typing import List, Dict, Union, Callable
import string
import six

import nltk
from nltk.stem import WordNetLemmatizer as lemma
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from datasets.tokenization import FullTokenizer

nltk.download("wordnet")
nltk.download('punkt')


class Preprocesser(object):
    def __init__(self, ) -> None:
        self.stopwords = set(stopwords.words("english"))

    def lower_case(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        text = text.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        return text

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [lemma().lemmatize(word=w, pos="v") for w in tokens]

    def clean_stopwords(self, tokens: List[str]) -> List[str]:
        return [item for item in tokens if item not in self.stopwords]

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    @staticmethod
    def convert_to_bert_input(text: str, max_length: int,
                              tokenizer: FullTokenizer,
                              cls: bool) -> Union[int, List[int]]:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length - 2:
            tokens = tokens[:max_length - 2]
        if cls:
            tokens = ["[CLS]"] + tokens
        tokens += ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        return input_ids

    @staticmethod
    def convert_to_unicode(text):
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")


class Corpus(Preprocesser):
    def __init__(self,
                 corpus: Dict[int, str],
                 do_sw: bool = True,
                 lemmatize: bool = True) -> None:
        super().__init__()
        self.corpus = corpus
        self.pids = list(corpus.keys())
        self.tokens = self.preprocess(do_sw, lemmatize)

    def __len__(self) -> int:
        return len(self.corpus)

    def preprocess(self, do_sw: bool, lemmatize: bool) -> Dict[int, List[str]]:
        docs_tokens: Dict[int, List[str]] = {}
        for pid, doc in self.corpus.items():
            doc = self.remove_punctuation(self.lower_case(doc))
            doc_tokened = self.tokenize(doc)
            if do_sw: doc_tokened = self.clean_stopwords(doc_tokened)
            if lemmatize: doc_tokened = self.lemmatize(doc_tokened)
            docs_tokens[pid] = doc_tokened
        return docs_tokens


class Query(Preprocesser):
    def __init__(self,
                 qid: int,
                 query: str,
                 do_sw: bool = True,
                 lemmatize: bool = True) -> None:
        super().__init__()
        self.query = Query.convert_to_unicode(query)
        self.qid = qid
        self.answers: Dict[int, float] = {}
        self.tokens = self.preprocess(do_sw, lemmatize)

    def preprocess(self, do_sw: bool, lemmatize: bool) -> List[str]:
        query = self.remove_punctuation(self.lower_case(self.query))
        query_tokened = self.tokenize(query)
        if do_sw: query_tokened = self.clean_stopwords(query_tokened)
        if lemmatize: query_tokened = self.lemmatize(query_tokened)
        return query_tokened

    def update_answers(self, new_answers: Dict[int, float], n: int):
        if not self.answers:
            self.answers = new_answers
        else:
            temp = {**self.answers, **new_answers}
            new = dict(sorted(temp.items(), key=lambda item: item[1]))
            firstpairs = {k: new[k] for k in list(new)[:n]}
            self.answers = firstpairs


if __name__ == "__main__":
    corpus = {
        1: "I love pudding",
        2: "I go to the beach",
        3: "the weather is nice today"
    }
    query = {3000: "What is love ?"}
    quer = Query(list(query.keys())[0], query[list(query.keys())[0]])
    corp = Corpus(corpus)
    print("corpus:", corp.tokens)
    print("query", quer.tokens)
