import os
import argparse

import tensorflow as tf

from src.bert import Bert
from datasets.utils import check_file_exists

BERT_MODEL = "bert_en_uncased_L-12_H-768_A-12"

parser = argparse.ArgumentParser(description="BERT Training parameters")
parser.add_argument("--data_dir",
                    type=str,
                    default=os.path.join("datasets", "datasets", "tfrecord"))
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--decay_steps", type=int, default=30000)
parser.add_argument("--warmup_steps", type=int, default=3000)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=3)

args = parser.parse_args()
bert_url = f"https://tfhub.dev/tensorflow/{BERT_MODEL}/2"


@tf.function
def extract_fn(record):
    features = {
        "query_ids":
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "doc_ids":
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "label":
            tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(record, features)


def format_fn(record):
    query_ids = tf.cast(record['query_ids'], tf.int32)
    doc_ids = tf.cast(record['doc_ids'], tf.int32)
    label_id = tf.cast(record['label'], tf.int32)
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

    return x, label_id


def load_data(record_path: str, bs: int, max_length: int) -> tf.data.Dataset:
    data = tf.data.TFRecordDataset(record_path)
    data = data.shuffle(100)
    data = data.repeat()
    data = data.map(extract_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.map(format_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    data = pad_dataset(data, bs, max_length)
    return data


def pad_dataset(data: tf.data.Dataset, bs: int, max_length: int):
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


def main():
    path = os.path.join(args.data_dir, "train_ds.tf")
    if not check_file_exists(path):
        raise ValueError("the path to the file specified doesn't exist")
    dataset = load_data(path, args.batch_size, args.max_seq_length)
    bert = Bert(bert_url, args.max_seq_length)
    print("building model...")
    model = bert.build_model(num_labels=1,
                             lr=args.learning_rate,
                             decay_steps=args.decay_steps,
                             warmup_steps=args.warmup_steps,
                             weight_decay=args.weight_decay)
    print(model.summary())
    model.fit(dataset, epochs=args.epochs, verbose=1, steps_per_epoch=400000)


if __name__ == "__main__":
    main()
