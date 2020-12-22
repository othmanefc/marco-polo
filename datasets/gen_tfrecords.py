import argparse
import collections
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub

from datasets import tokenization
from datasets.utils import check_file_exists, create_dir
from src.preprocesser import Preprocesser

HUB_PATH = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
parser = argparse.ArgumentParser(
    description="Training set creation parameters")
parser.add_argument("--output_folder",
                    default="datasets/datasets/tfrecord",
                    help="the output folder",
                    type=str)
parser.add_argument("--train_ds_path",
                    default="datasets/datasets/triples.train.small.tsv",
                    help="path to training dataset",
                    type=str)
parser.add_argument("--bert_model_hub",
                    help="Bert Model from HUB",
                    default=HUB_PATH,
                    type=str)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=256,
)
parser.add_argument("--max_train_examples",
                    default=None,
                    help="number of observations to be processed",
                    type=int)
args = parser.parse_args()


def write_to_tf_record(writer,
                       tokenizer,
                       query,
                       docs,
                       labels,
                       ids_file=None,
                       query_id=None,
                       doc_ids=None):
    query = Preprocesser.convert_to_unicode(query)
    query_token_ids = Preprocesser.convert_to_bert_input(
        query, max_length=args.max_seq_length, tokenizer=tokenizer, cls=True)
    q_t_ids_tf = tf.train.Feature(int64_list=tf.train.Int64List(
        value=query_token_ids))
    assert len(docs) == len(labels)
    for i, (doc_text, label) in enumerate(zip(docs, labels)):
        doc_text = Preprocesser.convert_to_unicode(doc_text)
        doc_token_id = Preprocesser.convert_to_bert_input(
            text=doc_text,
            max_length=args.max_seq_length - len(query_token_ids),
            tokenizer=tokenizer,
            cls=False)
        doc_ids_tf = tf.train.Feature(int64_list=tf.train.Int64List(
            value=doc_token_id))
        labels_tf = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[label]))
        features = tf.train.Features(feature={
            "query_ids": q_t_ids_tf,
            "doc_ids": doc_ids_tf,
            "label": labels_tf
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

        if ids_file:
            ids_file.write("\t".join([query_id, doc_ids[i]]) + "\n")


def convert_train_ds(tokenizer):
    if not check_file_exists(args.train_ds_path):
        raise ValueError(
            "train set not found please download it or specify right folder..")
    print('Converting Train to TfRecord...')
    print('Counting number of examples...')
    num_lines = sum(1 for _ in open(args.train_ds_path))
    print(f"{num_lines} Samples found...")
    writer = tf.io.TFRecordWriter(args.output_folder + "/train_ds.tf")
    max_train_examples = num_lines
    if args.max_train_examples:
        max_train_examples = min(max_train_examples, args.max_train_examples)
    print("Only processing {} % of the training set...".format(
        100 * max_train_examples / num_lines))
    with open(args.train_ds_path, "r") as file:
        pbar = tqdm(enumerate(file), total=num_lines)
        for i, line in pbar:
            if i > max_train_examples:
                print("Limit reached...")
                break
            query, pos_doc, neg_doc = line.rstrip().split('\t')
            write_to_tf_record(writer=writer,
                               tokenizer=tokenizer,
                               query=query,
                               docs=[pos_doc, neg_doc],
                               labels=[1, 0])
    writer.close()


def convert_eval_ds(tokenizer):
    if not check_file_exists(args.eval_ds_path):
        raise ValueError(
            "train set not found please download it or specify right folder")
    print('Converting Eval to TfRecord...')
    queries_docs = collections.defaultdict(list)
    query_map = {}
    eval_file = open(args.eval_ds_path, 'r')
    for i, line in enumerate(eval_file):
        query_id, doc_id, query, doc = line.strip().split('\t')
        label = 0  # Placeholder
        queries_docs[query_id].append([doc_id, doc, label])
        query_map[query_id] = query
    eval_file.close()
    for idd, doc in queries_docs.items():
        queries_docs[idd] = doc + max(
            0, args.num_eval_docs - len(doc)) * [('000000000', 'FAKE DOC', 0)]
    writer = tf.io.TFRecordWriter(args.output_folder + "/eval_ds.tf")
    map_ids_file = open(args.output_folder + "/query_doc_ids_eval.txt")
    for i, (query_id, docs) in tqdm(enumerate(queries_docs.items())):
        doc_ids, docs, labels = zip(*docs)
        query = query_map[query_id]
        write_to_tf_record(writer=writer,
                           tokenizer=tokenizer,
                           query=query,
                           docs=docs,
                           labels=labels,
                           ids_file=map_ids_file,
                           query_id=query_id,
                           doc_ids=doc_ids)
    map_ids_file.close()
    writer.close()


if __name__ == "__main__":
    print("downloading layer...")
    bert_layer = hub.KerasLayer(args.bert_model_hub, trainable=True)
    print("Layer downloaded...")
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    create_dir(args.output_folder)
    convert_train_ds(tokenizer)
    convert_eval_ds(tokenizer)
