import logging
import os
from tqdm.asyncio import tqdm
import asyncio

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from src.dataloader import DataLoader

COLLECTION_PATH = os.environ.get(
    "COLLECTION_PATH",
    os.path.join("datasets", "datasets", "collections", "collection.tsv"))
es = AsyncElasticsearch()

logging.basicConfig(level=logging.INFO)


async def map_doc(index_name="documents"):
    CHUNKSIZE = 50000
    document_loader = DataLoader(COLLECTION_PATH,
                                 chunksize=CHUNKSIZE,
                                 names=["pid", "passage"])
    for _, collection in tqdm(enumerate(document_loader.reader),
                              desc="collection"):
        for i, row in collection.iterrows():
            yield {
                '_op_type': 'index',
                '_index': index_name,
                'pid': row.pid,
                'passage': row.passage,
            }


async def main():
    await async_bulk(es, map_doc())


if __name__ == "__main__":
    logging.info("Starting indexation of documents...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    logging.info("indexation done...")
