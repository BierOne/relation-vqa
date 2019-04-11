import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn
import utils.config as config

class TextProcessor(nn.Module):
	def __init__(self, embeddings, embedding_features, gru_features, drop=0.0):
		super(TextProcessor, self).__init__()
		self.drop = nn.Dropout(drop)
		self.tanh = nn.Tanh()
		self.gru = nn.GRU(input_size=embedding_features,
						   hidden_size=gru_features,
						   num_layers=1,
						   batch_first=True)

		self._init_gru(self.gru.weight_ih_l0)
		self._init_gru(self.gru.weight_hh_l0)
		self.gru.bias_ih_l0.data.zero_()
		self.gru.bias_hh_l0.data.zero_()

		if config.pretrained_model == 'glove':
			self.embedding = Glove(embeddings, fine_tune=False)
		else:
			self.embedding = nn.Embedding(
				embeddings, embedding_features) # , padding_idx=0)
			nn.init.xavier_uniform_(self.embedding.weight)

	def _init_gru(self, weight):
		for w in weight.chunk(3, 0):
			nn.init.xavier_uniform_(w)

	def forward(self, q, q_len):
		embedded = self.embedding(q)
		tanhed = self.tanh(self.drop(embedded))
		packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
		_, h = self.gru(packed)
		return h.squeeze(0)


class FactExtractor(nn.Sequential):
	def __init__(self, embeddings):
		super(FactExtractor, self).__init__()
		self.add_module('embedding',  Glove(embeddings, fine_tune=False))
		
		
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