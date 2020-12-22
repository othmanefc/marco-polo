import os
import json
import logging

from elasticsearch import Elasticsearch

logger = logging.getLogger()
logger.setLevel("INFO")


def main():
    es = Elasticsearch()
    logger.info("Deleting index if already existing...")
    es.indices.delete(index="documents", ignore=[404])
    with open(os.path.join("src", "elasticsearch", "index.json"),
              encoding='utf-8',
              errors='ignore') as index_file:
        index_map = json.load(index_file)
        logger.info("Creating index...")
        es.indices.create(index="documents", body=index_map)
        logger.info("Index created...")


if __name__ == '__main__':
    main()
