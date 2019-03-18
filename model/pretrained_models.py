import sys
import torch
import torch.nn as nn

# from pytorch_pretrained_bert import BertTokenizer
# from pytorch_pretrained_bert import BertModel


class Bert(object):
	""" Mimicing the tutorial of the re-implemented pytorch Bert model (https://arxiv.org/abs/1810.04805).
		The referred examples are from
		https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py.
		max_len is the maximum length of processed questions, thus the max_len for bert model is 
		max_len + 2 (including '[CLS]' and '[SEP]'), according to the original paper.
	"""
	def __init__(self, bert_model, max_len, tokenize_only=False):
		if tokenize_only:
			self.max_len = max_len
			self.tokenizer = BertTokenizer.from_pretrained(bert_model)
		else:
			self.model = BertModel.from_pretrained(bert_model).cuda()
			self.model.eval()
			self.model = nn.DataParallel(self.model)

	def tokenize_text(self, question_text):
		tokenized_text = self.tokenizer.tokenize(question_text)
		if len(tokenized_text) > self.max_len: # truncate question length
			tokenized_text = tokenized_text[:self.max_len]
		tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']

		input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

		# the mask has 1 for real tokens and 0 for padding tokens
		input_mask = [1] * len(input_ids)

		# zero-pad up to the max length ('[CLS]' and '[SEP]')
		padding = [0] * (self.max_len+2-len(input_ids))
		input_ids += padding
		input_mask += padding

		assert len(input_ids) == self.max_len + 2
		assert len(input_mask) == self.max_len + 2
		
		input_ids = torch.tensor(input_ids)
		input_mask = torch.tensor(input_mask)

		return input_ids, input_mask

	def forward(self, input_ids, input_mask):
		""" Since there is only one sentence, we remove the token_type_ids which is responsible for 
			sentence type. We only extract the features from the last attention block, which means 
			output_all_encoded_layers should be set to False.
		"""
		features, _ = self.model(input_ids=input_ids, 
								attention_mask=input_mask, 
								output_all_encoded_layers=False)
		return features


class Glove(nn.Module):
	def __init__(self, weights, fine_tune=True):
		super(Glove, self).__init__()
		num_embeddings = weights.shape[0]
		embedding_dim = weights.shape[1]
		weights = torch.tensor(weights)

		self.embeddings = nn.Embedding(
			num_embeddings, embedding_dim, padding_idx=0)
		self.embeddings.load_state_dict({'weight': weights})
		if not fine_tune:
			self.embeddings.weight.requires_grad = False

	def forward(self, idx):
		# if idx.shape[0]>1:
			# return torch.cat()
		return self.embeddings(idx)



