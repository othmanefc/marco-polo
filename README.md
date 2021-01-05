# Information retrieval using BERT and MS Marco Datasets
> by Othmane Hassani & Antoine Lajoinie

## Prerequisites

- Docker

- Docker Compose

- Virtualenv


## Training Model from MS Marco Task

You can train a model to accomplish MS Marco task:

### Create the datasets

You need to create a virtualenv:

```bash 
$ cd path/to/marco-polo
$ python3 -m venv env-marco
$ source env-marco/bin/activate/
$ pip3 install -r requirements.txt
```
Run the following commands you just need to be in the main directory of the repository

```bash
$ cd path/to/marco-polo
$ DATA_DIR=./datasets/datasets
$ mkdir ${DATA_DIR}
$ wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz -P ${DATA_DIR}
$ wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz -P ${DATA_DIR} 

$ tar -xvf ${DATA_DIR}/triples.train.small.tar.gz -C ${DATA_DIR}
$ tar -xvf ${DATA_DIR}/top1000.dev.tar.gz -C ${DATA_DIR}
$ tar -xvf ${DATA_DIR}/top1000.eval.tar.gz -C ${DATA_DIR}
```

You then convert our dataset into a TFRecord dataset which is a Binary Tensorflow Data Class, it is better to have the data this way, as the data size is really huge and this type of dataset is optimized.
We will create a training set from the `triples.train.small.tsv`. Each observation correspond to a query, a passage to corresponding to it positively, and one negatively. So, we split that into two observation and we do that for each triplet, until we reach the `max_train_examples` argument (default to 500000). Then we will write the eval dataset which comes from `top1000.eval.tsv`. With the `num_eval_docs` argument (default to 10), we can specify the number of documents per query to keep.
```bash
$ python -m datasets.gen_tfrecords \
        --output_folder=${DATA_DIR}/tfrecord \
        --train_ds_path=${DATA_DIR}/triples.train.small.tsv \
        --eval_ds_path=${DATA_DIR}/top1000.eval.tsv
        --max_seq_length=512
```
And if you prefer, you can also download the tfrecord files yourself, from [here](www.kaggle.com/dataset/7d81e21833a9844c5434e40fc51b25d9c2b7f6fb2e823052e9c3a55150b4251d) and copy the files to the specified datasets.

### Download pretrained model

You need to download a pretrained BERT model, you can find [here](https://github.com/google-research/bert). If you have a good enough computer and GPU, we will use Bert Base


```bash
$ wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
$ unzip uncased_L-12_H-768_A-12.zip
```

### Fine-tune the model

Afterward, you can fine-tune the model using either the [Kaggle](https://www.kaggle.com/lemartiens/ranking-msmarco/notebook) or locally with scripts available. We have many parameters here:

`learning rate` : (default to 1e-5) This default value is generally the standard for BERT tasks, you can maybe try 2e-5 or 3e-5 too.
`batch_size`: (default to 8) A higher batch size would probably not fit into memory.
`max_seq_length`: (default to 512) This parameter can also be lower depending on the GPU available for training.
`decay_steps`: (default to 30000) Number of steps where the learning rate will decay.
`warmup_steps`: (default to 3000) Number of steps where the learning rate will increase to the specified learning rate from 0.
`weight_decay`: (default to 1e-3) This paramater is an additional term in the weight update rule that causes the weights to exponentially decay to zero.
`epochs`: (default to 3) You can probably only use one epoch as the data feeding is shuffled and repeated.
`steps_per_epochs`: (default to 40000) Number of training steps per epoch.
`top_n`: (default to 10) Number of eval docs per query to keep (should be kept at 10, since we use MRR@10).

**Fine-tune the model:**

```bash
$ python3 -m src.train.train_bert --data_dir=/path/to/tfrecord_folder \
                                --output_dir=/path/to/outputdir \
                                --learning_rate=1e-5 \
                                --batch_size=8 \
                                --max_seq_length=512 \
                                --decay_steps=30000 \
                                --warmup_steps=3000 \
                                --weight_decay=1e-3 \
                                --epochs=3 \
                                --steps_per_epoch=40000 \
                                --do_train=True \
                                --do_eval=False \
                                --top_n=10
```
You can do use the model to submit to MS Marco, which I haven't done and wasn't the purpose of this task.

## Running the APP

For this part of the project, We will use a BERT Large Finetuned on SQUAD which is really good for Question answering task.

### Build Docker

Now, you need to build the Docker, it should build an ElasticSearch and a Flask image.

```bash
$ docker-compose up
```

### Index documents

You need to index documents onto the elasticsearch cluster. So, first create the index, it will use the mapping for `src/elasticsearch/index.json`. You can change the number of shards or replicas if needed:

```bash
$ python3 src/elasticsearch/src/create_index.py
```

It will create an index names "documents" to store our documents from MSMarco dataset, if you haven't download it, do it now !

```bash
$ wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -P ./datasets/datasets/collections
$ tar -xvf ./datasets/datasets/collections/collection.tar.gz -C ./datasets/datasets/collections/collection.tar.gz
$ export COLLECTION_PATH=./datasets/datasets/collections/collection.tar.gz
```
Then, index them:

```bash 
$ python3 -m src.elasticsearch.feed
```
As it can take a long time to index all the documents, you can go to the next part while it's still processing.


#### Flask

Last thing is to start the app, which you can do using the following command:

```bash
$ python3 -m web.app_fl
```

By default the app should be at: http://localhost:8300/

An example below:

![example](./assets/example.png)

#### How it works
The app will use the following workflow:
 - `Elasticsearch` It has a very powerful search engine, based on different algorithms. For this case we will use the implementation of BM25 that will return the n (10 here) documents that match the question.
 ```python
def query(question, var='passage', size=10):
    logging.info('Searching on ES cluster...')
    script_params = {"match": {var: question}}
    docs = es.search(index="documents",
                     body={
                         "size": size,
                         "query": script_params,
                         "_source": {
                             "includes": ["pid", "passage"]
                         }
                     })
    logging.info('Found documents...')
    return parse_results(docs, size)
```

 - `BERT` The return documents from ElasticSearch will then be inputted into a BERT Large SQUAD finetuned. But first, they will of course be tokenized, the only truncated part is the passage. Maybe we could improve the tokenization on this case. Then we will create a `confidence` value computed from the `start_score` and `end_score`. The 10 documents will then be ranked through a softmax transformation then ranking.


 - `WEB` The results of the model will then be parsed into a webpage using Flask and a simple VUE page.

#### Streamlit

You can also run the app with Streamlit which is more memory hogging than Flask

```bash
$ streamlit run app.py
```