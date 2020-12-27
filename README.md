# Information retrieval using BERT and MS Marco Datasets

## Prerequisites

- Docker

- Docker Compose

## How to Use

First, you need to download a pretrained BERT model, you can find [here](https://github.com/google-research/bert). If you have a good enough computer and GPU, we will use Bert Base

### 1. Download Pretrained BERT

```bash
$ wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
$ unzip uncased_L-12_H-768_A-12.zip
```

### 2. Fine-tune the model

Afterward, you can fine-tune the model using either the [Kaggle](https://www.kaggle.com/lemartiens/ranking-msmarco/notebook) or locally with scripts available

If you wish to train locally, you need to process the datasets, or download the TFRecord [here](www.kaggle.com/dataset/7d81e21833a9844c5434e40fc51b25d9c2b7f6fb2e823052e9c3a55150b4251d)

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

### 3. Set environment variables

Set environment variables for BERT pre-trained model and fine-tuned one:

```bash
$ export PATH_MODEL=./path/to/pretrained_model
$ export PATH_TUNED=./path/to/finetuned_model
```

### 4. Build Docker

Now, you need to build the Docker, it should build an ElasticSearch and a Bert-as-service images for the streamlit app

```bash
$ sudo -E docker-compose up
```

### 5. Index documents

You need to index documents onto the elasticsearch cluster. So, first create the index, it will use the mapping for src/elasticsearch/index.json, specify the right length for the dense vector depending on the size of the BERT Model used:

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

### 6. Run the App

Last thing is to start the app, which you can do using the following command

```bash
$ streamlit run app.py
```

By default the app should be on: http://localhost:8501/

