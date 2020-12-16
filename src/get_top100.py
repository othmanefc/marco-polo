import os
from typing import Union, List, Dict

from tqdm import tqdm
import pandas as pd

from bm25 import BM25
from preprocesser import Query, Corpus
from dataloader import DataLoader

QUERIES_PATH = os.environ.get(
    "QUERIES_PATH", os.path.join("datasets", "queries", "queries.train.tsv"))

COLLECTION_PATH = os.environ.get(
    "COLLECTION_PATH", os.path.join("datasets", "collection",
                                    "collection.tsv"))

OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH_BM25", os.path.join("datasets", "bm25", "bm25.train.tsv"))

TOP_N = 50


def train_bm25(queries: List[Query], collection: Corpus) -> List[Query]:
    bm25 = BM25(collection)
    queries_list: List[Query] = []
    for query in queries:
        top_10 = bm25.top_n(query, n=TOP_N)
        query.update_answers(top_10, n=TOP_N)
        queries_list.append(query)
    return queries_list


def save_top10(result):
    result.to_csv(OUTPUT_PATH, chunksize=500, index=False)


def reshape_df(
        queries: pd.DataFrame = None,
        collection: pd.DataFrame = None) -> Union[List[Query], Corpus, None]:
    if queries is not None:
        return [
            Query(qid, query)
            for qid, query in zip(queries["qid"], queries["query"])
        ]
    elif collection is not None:
        return Corpus(dict(zip(collection["pid"], collection["passage"])))
    else:
        return None


def main():
    queries_loader = DataLoader(QUERIES_PATH,
                                chunksize=500,
                                names=["qid", "query"])
    collection_loader = DataLoader(COLLECTION_PATH,
                                   chunksize=10000,
                                   names=["pid", "passage"])
    result: Dict[int, Dict[int, float]] = {}
    for queries in tqdm(queries_loader.reader, desc="Queries"):
        queries = reshape_df(queries=queries)
        for collection in tqdm(collection_loader.reader, desc="collection"):
            collection = reshape_df(collection=collection)
            queries = train_bm25(queries, collection)
        result = dict(result,
                      **{query.qid: query.answers
                         for query in queries})


if __name__ == "__main__":
    main()
