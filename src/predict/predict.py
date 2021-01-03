import logging

import numpy as np
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer
from elasticsearch import Elasticsearch

es = Elasticsearch()
logging.basicConfig(level=logging.INFO)


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


def parse_results(json, size):
    results = json['hits']['hits']
    assert len(results) == size
    for hit in results:
        yield hit['_source']


class Predict:

    def __init__(
            self,
            model_name='bert-large-uncased-whole-word-masking-finetuned-squad'
    ):
        self.model_name = model_name
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(
            self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _reconstruct_text(self, tokens, start=0, stop=-1):
        tokens = tokens[start:stop]
        txt = ' '.join(tokens)
        txt = txt.replace('[SEP]', '')
        txt = txt.replace('[PAD]', '')
        txt = txt.replace(' ##', '')
        txt = txt.replace('##', '')
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(' .', '.')
        txt = txt.replace('( ', '(')
        txt = txt.replace(' )', ')')
        txt = txt.replace(' - ', '-')
        txt_list = txt.split(' , ')
        txt = ''
        length = len(txt_list)
        if length == 1:
            return txt_list[0]
        new_list = []
        for i, t in enumerate(txt_list):
            if i < length - 1:
                if t[-1].isdigit() and txt_list[i + 1][0].isdigit():
                    new_list += [t, ',']
                else:
                    new_list += [t, ', ']
            else:
                new_list += [t]
        return ''.join(new_list)

    def predict_batch(self, question, passages):
        seqs = [[question, doc] for doc in passages]
        logging.info("Encoding sequences...")
        batch = self.tokenizer.batch_encode_plus(seqs,
                                                 return_tensors='tf',
                                                 max_length=512,
                                                 truncation='only_second',
                                                 padding=True)
        logging.info('Enconding done...')
        tokens_batch = list(
            map(self.tokenizer.convert_ids_to_tokens, batch['input_ids']))
        logging.info('outputting from model...')
        model_output = self.model(batch['input_ids'], batch['attention_mask'],
                                  batch['token_type_ids'])
        start_scores, end_scores = model_output[0], model_output[1]
        logging.info('outputting done...')
        start_scores = start_scores[:, 1:-1]  # skipping SEP and CLS
        end_scores = end_scores[:, 1:-1]
        answer_starts = np.argmax(start_scores, axis=1)
        answer_ends = np.argmax(end_scores, axis=1)

        assert len(tokens_batch) == len(answer_starts)
        answers = []
        logging.info('parsing results from predictions...')
        for i, tokens in enumerate(tokens_batch):
            answer_start, answer_end = answer_starts[i], answer_ends[i]
            full_txt = passages[i]
            answer = self._reconstruct_text(tokens, answer_start,
                                            answer_end + 2)
            ans = {
                'answer': answer,
                'start': answer_start,
                'end': answer_end,
                'full_context': full_txt
            }
            ans['confidence'] = (start_scores[i, answer_start] +
                                 end_scores[i, answer_end]).numpy()
            answers.append(ans)
        return answers

    def query(self, question, size=10):
        return list(query(question, size=size))

    def search(self, question, n_answers=10, batch_size=8, size_query=50):
        doc_results = self.query(question, size=size_query)
        print(doc_results)
        if len(doc_results) == 0:
            return []

        passages = []
        for doc in doc_results:
            passages.append(doc.get('passage'))
        if batch_size > len(passages):
            batch_size = len(passages)
        num_chunks = (len(passages) // batch_size) + 1
        batches = Predict._chunks(passages, num_chunks)
        answers = []
        for i, batch in enumerate(batches):
            logging.info(f'Processing batch {i+1}...')
            predictions = self.predict_batch(question, batch)
            for ans in predictions:
                answers.append(ans)
        answers = sorted(answers,
                         key=lambda ans: ans['confidence'],
                         reverse=True)[:n_answers]
        confs = [ans.get('confidence') for ans in answers]
        max_conf = max(confs)
        exp_scores = []
        for c in confs:
            exp_scores.append(np.exp(c - max_conf))
        total = sum(exp_scores)
        for i, _ in enumerate(confs):
            answers[i]['confidence'] = exp_scores[i] / total
        return answers

    @staticmethod
    def _chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
