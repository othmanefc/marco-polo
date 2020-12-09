# How to download the datasets

Run the following commands you just need to be in the main directory of the repository

```bash
cd path/to/marco-polo
DATA_DIR=./datasets/datasets
mkdir ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz -P ${DATA_DIR}
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz -P ${DATA_DIR} 

tar -xvf ${DATA_DIR}/triples.train.small.tar.gz -C ${DATA_DIR}
tar -xvf ${DATA_DIR}/top1000.dev.tar.gz -C ${DATA_DIR}
tar -xvf ${DATA_DIR}/top1000.eval.tar.gz -C ${DATA_DIR}
```
If you want to train the model yourself, you also need to convert the downloaded datasets into a TFRecord for proper training

```bash
python -m datasets.train_ds \
        --output_folder=${DATA_DIR}/tfrecord \
        --train_ds_path=${DATA_DIR}/triples.train.small.tsv \
        --max_seq_length=256
```

And if you prefer, you can also download the tfrecord files yourself, from [here](https://drive.google.com/file/d/1IHFMLOMf2WqeQ0TuZx_j3_sf1Z0fc2-6/view) and copy the files to the specified datasets