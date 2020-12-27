#!/bin/sh
bert-serving-start -http_port 8125 -num_worker=1 -max_seq_len=256 -model_dir /model --tuned_model_dir /tuned_model