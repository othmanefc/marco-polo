import unittest

from src.predict import Predict


class TestPredict(unittest.TestCase):
    pred = Predict(
        model_name='bert-large-uncased-whole-word-masking-finetuned-squad')
    question = ['what do I love ?']
    answers = ['I love food', 'I drink water']

    def test__reconstruct_text(self):
        seqs = [[self.question[0], ans] for ans in self.answers]
        batch = self.pred.tokenizer.batch_encode_plus(seqs,
                                                      return_tensors='tf',
                                                      max_length=128,
                                                      truncation='only_second',
                                                      padding=True)
        tokens_batch = list(
            map(self.pred.tokenizer.convert_ids_to_tokens, batch['input_ids']))
        for i, tokens in enumerate(tokens_batch):
            rec = self.pred._reconstruct_text(tokens)
            seq_str = " ".join(seqs[i])
            seq_str = "[CLS] " + seq_str
            self.assertEqual(seq_str, rec)
            self.assertTrue(all(isinstance(s, str) for s in rec))

    def test_predict_batch(self):
        self.assertIsNone(self.pred.predict_batch('', []))
        predictions = self.pred.predict_batch(self.question[0], self.answers)
        for pred in predictions:
            self.assertTrue(
                all(bb in list(pred.keys()) for bb in
                    ['confidence', 'full_context', 'start', 'end', 'answer']))
            self.assertIsInstance(pred['confidence'], float)
            self.assertIsInstance(pred['full_context'], str)
            self.assertIsInstance(pred['start'], int)
            self.assertIsInstance(pred['end'], int)
            self.assertTrue(pred['answer'] in pred['full_context'])
