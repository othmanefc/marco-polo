from typing import List
import os
import argparse
import logging
import _io
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from src.train.bert import Bert
from src.train.metrics import mrr
from datasets.utils import check_file_exists

BERT_MODEL = "bert_en_uncased_L-12_H-768_A-12"

parser = argparse.ArgumentParser(description="BERT Training parameters")
parser.add_argument("--data_dir",
                    type=str,
                    default=os.path.join("datasets", "datasets", "tfrecord"))
parser.add_argument("--output_dir", type=str, default=os.path.join("datasets", "datasets", "outputs"))
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--decay_steps", type=int, default=30000)
parser.add_argument("--warmup_steps", type=int, default=3000)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--steps_per_epoch", type=int, default=40000)
parser.add_argument("--do_train", type=bool, default=True)
parser.add_argument("--do_eval", type=bool, default=True)
parser.add_argument("--top_n", type=int, default=10)

args = parser.parse_args()
logger = logging.getLogger()
logger.setLevel("INFO")
bert_url = f"https://tfhub.dev/tensorflow/{BERT_MODEL}/2"


@tf.function
def extract_fn(record, train: bool = True):
    if train:
        features = {
            "query_ids":
                tf.io.FixedLenSequenceFeature([], tf.int64,
                                              allow_missing=True),
            "doc_ids":
                tf.io.FixedLenSequenceFeature([], tf.int64,
                                              allow_missing=True),
            "label":
                tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        features = {
            "query_ids":
                tf.io.FixedLenSequenceFeature([], tf.int64,
                                              allow_missing=True),
            "doc_ids":
                tf.io.FixedLenSequenceFeature([], tf.int64,
                                              allow_missing=True),
        }

    return tf.io.parse_single_example(record, features)


@tf.function
def format_fn(record, train: bool = True):
    query_ids = tf.cast(record['query_ids'], tf.int32)
    doc_ids = tf.cast(record['doc_ids'], tf.int32)
    input_ids = tf.concat((query_ids, doc_ids), 0)
    query_segment_id = tf.zeros_like(query_ids)
    doc_segment_id = tf.ones_like(doc_ids)
    segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)
    input_mask = tf.ones_like(input_ids)

    x = {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
        "input_mask": input_mask
    }
    label_id = tf.cast(record['label'], tf.int32)
    return x, label_id


@tf.function
def cut_if_longer(x, y, train: bool, max_length: int):
    new_x = x.copy()
    for id, val in x.items():
        if tf.math.greater(tf.shape(val), max_length):
            new_x[id] = tf.slice(val, begin=[0], size=[max_length])
    return new_x, y


def load_data(record_path: str,
              bs: int,
              max_length: int,
              train: bool,
              max_eval_obs: int = 1000) -> tf.data.Dataset:
    data = tf.data.TFRecordDataset(record_path)
    data = data.map(lambda x: extract_fn(x, train),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.map(lambda x: format_fn(x, train),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.map(lambda x, y: cut_if_longer(x, y, train, max_length),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if train:
        data = data.shuffle(100)
        data = data.repeat()
    else:
        data = data.take(max_eval_obs * 1000)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    data = pad_dataset(data, bs, max_length, train)
    return data


def pad_dataset(data: tf.data.Dataset, bs: int, max_length: int, train: bool):
    data = data.padded_batch(batch_size=bs,
                             padded_shapes=({
                                 "input_ids": [max_length],
                                 "segment_ids": [max_length],
                                 "input_mask": [max_length]
                             }, []),
                             padding_values=({
                                 "input_ids": 0,
                                 "segment_ids": 0,
                                 "input_mask": 0,
                             }, 0),
                             drop_remainder=True)
    return data


def msmarco_write(query_ids: List[int], doc_ids: List[int], preds: List[int],
                  msmarco_file: _io.TextIOWrapper):
    assert len(set(query_ids)) == 1
    query_id = query_ids[0]
    rank = 1
    logger.info("writing to MSMarco file...")
    for idx in preds:
        doc_id = doc_ids[idx]
        msmarco_file.write("\t".join((str(query_id), str(doc_id), str(rank))) +
                           "\n")
        rank += 1


def main():
    path = os.path.join(args.data_dir, "train_ds.tf")
    if not check_file_exists(path):
        raise ValueError("the path to the file specified doesn't exist")
    train_dataset = load_data(path, args.batch_size, args.max_seq_length, True)
    bert = Bert(bert_url, args.max_seq_length)
    print("building model...")
    model = bert.build_model(num_labels=1,
                             lr=args.learning_rate,
                             decay_steps=args.decay_steps,
                             warmup_steps=args.warmup_steps,
                             weight_decay=args.weight_decay)
    print(model.summary())
    if args.do_train:
        logger.info("Training...")
        model.fit(train_dataset,
                  epochs=args.epochs,
                  verbose=1,
                  steps_per_epoch=args.steps_per_epoch,
                  callbacks=bert.callbacks())
    if args.do_eval:
        logger.info("Evaluating...")
        top_n = min(args.top_n, 1000)
        if top_n != args.top_n:
            logger.warn(
                f"TOP N too high,evaluation with default value of {top_n}...")
        test_path = os.path.join(args.data_dir, "eval_ds.tf")
        if not check_file_exists(test_path):
            raise ValueError("the path to the file specified doesn't exist")
        max_eval_obs = min(args.max_eval_obs, 5000)
        total_num = max_eval_obs * top_n
        test_dataset = load_data(test_path,
                                 args.batch_size,
                                 args.max_seq_length,
                                 train=False,
                                 max_eval_obs=max_eval_obs)
        with open(os.path.join(args.data_dir, "query_doc_ids_eval.txt"),
                  'r') as query_doc_map:
            query_map = [line.strip().split("\t") for line in query_doc_map]
        bert_preds = model.predict(test_dataset)
        assert len(bert_preds) == total_num
        metric = []
        logger.info("Evaluation metric for each query...")
        pbar = tqdm(range(max_eval_obs))

        for index in (pbar):
            real_values = query_map[index * 1000:(index * 1000) + top_n]
            preds_subset = bert_preds[index * 1000:(index + 1) * 1000]
            preds_top_n = preds_subset.argsort()[::-1][:top_n]
            real_top_n = real_values[:top_n]
            metric.append(mrr(real_top_n, preds_top_n))

            query_ids, doc_ids = set(*real_values)
            pbar.set_description(f"Query ID {str(query_ids[0])}")
            msmarco_file = open(
                os.path.join(args.output_dir, "ms_marco_eval.tsv"), "w+")
            msmarco_write(query_ids, doc_ids, preds_top_n, msmarco_file)

        metric = np.mean(metric)
        logger.info(f"MMR@10 => {metric}")
        msmarco_file.close()


if __name__ == "__main__":
    main()
