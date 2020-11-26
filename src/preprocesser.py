from typing import List, Dict, Union
import string

import nltk
from nltk.stem import WordNetLemmatizer as lemma
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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


class Corpus(Preprocesser):
    def __init__(self,
                 corpus: List[str],
                 do_sw: bool = True,
                 lemmatize: bool = True) -> None:
        super().__init__()
        self.corpus = corpus
        self.tokens = self.preprocess(do_sw, lemmatize)

    def __len__(self) -> int:
        return len(self.corpus)

    def preprocess(self, do_sw: bool, lemmatize: bool) -> List[List[str]]:
        docs_tokens = []
        for doc in self.corpus:
            doc = self.remove_punctuation(self.lower_case(doc))
            doc_tokened = self.tokenize(doc)
            if do_sw: doc_tokened = self.clean_stopwords(doc_tokened)
            if lemmatize: doc_tokened = self.lemmatize(doc_tokened)
            docs_tokens.append(doc_tokened)
        return docs_tokens


class Query(Preprocesser):
    def __init__(self,
                 query: str,
                 do_sw: bool = True,
                 lemmatize: bool = True) -> None:
        super().__init__()
        self.query = query
        self.tokens = self.preprocess(do_sw, lemmatize)

    def preprocess(self, do_sw: bool, lemmatize: bool) -> List[str]:
        query = self.remove_punctuation(self.lower_case(self.query))
        query_tokened = self.tokenize(query)
        if do_sw: query_tokened = self.clean_stopwords(query_tokened)
        if lemmatize: query_tokened = self.lemmatize(query_tokened)
        return query_tokened


if __name__ == "__main__":
    corpus = [
        "I love pudding", "I go to the beach", "the weather is nice today"
    ]
    query = "What is love ?"
    quer = Query(query)
    corp = Corpus(corpus)
    print("corpus:", corp.tokens)
    print("query", quer.tokens)
