import logging
import os
from tqdm import tqdm
import asyncio

from bert_serving.client import BertClient
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from src.dataloader import DataLoader

COLLECTION_PATH = os.environ.get(
    "COLLECTION_PATH",
    os.path.join("datasets", "datasets", "collections", "collection.tsv"))
es = AsyncElasticsearch()
bc = BertClient(output_fmt='list')

logger = logging.getLogger("bertclient")
logger.setLevel("INFO")


async def map_doc(index_name="documents"):
    document_loader = DataLoader(COLLECTION_PATH,
                                 chunksize=10000,
                                 names=["pid", "passage"])
    for collection in tqdm(document_loader.reader, desc="collection"):
        for i, row in tqdm(collection.iterrows(), total=10000):
            yield {
                '_op_type': 'index',
                '_index': index_name,
                'pid': row.pid,
                'passage': row.passage,
                'embedding': bc.encode([row.passage])
            }
        # with open('docs.jsonl', "w+") as doc_file:
        #     docs = []
        #     for i, row in collection.iterrows():
        #         mapp = map_doc(row['pid'], row['passage'], embeddings[i])
        #         docs.append(mapp)
        #         doc_file.write(json.dumps(mapp) + '\n')


async def main():
    await async_bulk(es, map_doc())


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
