import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.config as config
import model.counting as counting
from model.pretrained_models import Bert
from model.pretrained_models import Glove
from torch.nn.utils.rnn import pack_padded_sequence

class Net(nn.Module):
	def __init__(self, fact_embeddings=None):
		super(Net, self).__init__()
		text_features = 2400
		vision_features = config.output_features
		glimpses = 1
		objects = 100
		self.drop = nn.Dropout(0.5)
		self.tanh = nn.Tanh()
		self.lin = nn.Linear(text_features, vision_features)
		self.bn  = nn.BatchNorm1d(vision_features)
		
		self.ques_text = TextProcessor(
			embeddings=fact_embeddings,
			embedding_features=config.text_embed_size,
			gru_features=text_features,
			drop=0.5
		)
		self.fact_text = FactExtractor(
			embeddings=fact_embeddings,
		)
		self.qv_fusion = Mlb_with_attention(
			v_features=vision_features,
			q_features=text_features,
			mid_features=1024,
			drop=0.5,
		)
		self.fv_fusion = Mlb_with_attention(
			v_features=config.fact_embed_size,
			q_features=vision_features,
			mid_features=1024,
			drop=0.5,
		)
		self.classifier = Classifier(
			fv_features=glimpses * vision_features,
			fs_features=glimpses * config.fact_embed_size,
			mid_features=1024,
			out_features=config.max_answers,
			drop=0.5,
		)
		# self.counter = counting.Counter(objects)

		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, v, b, q, f, q_len):
		q = self.ques_text(q, q_len)
		v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v)
		# print(v.shape)
		n,c = f.shape[:2]
		f = self.fact_text(f).view(n, c, -1).permute(0, 2, 1).unsqueeze(2)
		f = f / (f.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(f)
		# print(f.shape)
		# question-visual attention
		fv = self.qv_fusion(v, q)
		fv = apply_attention(v, fv)
		# print(v.shape)
		fv = fv * self.tanh(self.bn(self.lin(self.drop(q))))
		# fact attention
		fs = self.fv_fusion(f, fv)
		fs = apply_attention(f, fs)
		
		answer = self.classifier(fv, fs)
		return answer


class Fusion(nn.Module):
	""" Crazy multi-modal fusion: negative squared difference minus relu'd sum
	"""
	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		# found through grad student descent ;)
		return - (x - y)**2 + F.relu(x + y)


class Mlb_with_attention(nn.Sequential):
	def __init__(self, v_features, q_features, mid_features, glimpses=1, drop=0.0):
		super(Mlb_with_attention, self).__init__() 
		self.drop = nn.Dropout(drop)
		self.tanh = nn.Tanh()
		self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)
		# self.lin11 = nn.Linear(v_features, mid_features, bias=False)
		self.lin12 = nn.Linear(q_features, mid_features, bias=False)
		self.x_conv = nn.Conv2d(mid_features, glimpses, 1)
		# self.lin2 = nn.Linear(mid_features, mid_features)
		self.lin3 = nn.Linear(mid_features, v_features)
		# self.lin32 = nn.Linear(q_features, v_features)
		self.bn11 = nn.BatchNorm2d(mid_features)
		self.bn12 = nn.BatchNorm1d(mid_features)
		# self.bn2 = nn.BatchNorm1d(mid_features)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x, y): # x -> (batchsize, featuremap, regions)
		# v = x.detach()
		# t = y.clone()
		# x = self.bn11(self.lin11(self.drop(x)))
		x =  self.bn11(self.v_conv(self.drop(x)))
		# print(x.shape)
		y = self.bn12(self.lin12(self.drop(y)))
		y = tile_2d_over_nd(y, x) # x->v, y->q
		x = self.tanh(x) * self.tanh(y)
		x = self.x_conv(self.drop(x))
		
		# x = self.lin2(self.drop(x))
		# x = self.softmax(self.lin3(x)).unsqueeze(1)
		# x = apply_attention(v, x)
		# x = x * tanh(self.lin32(t))
		
		return x

class Classifier(nn.Sequential):
	def __init__(self, fv_features, fs_features, mid_features, out_features, drop=0.0):
		super(Classifier, self).__init__()
		self.drop = nn.Dropout(drop)
		self.tanh = nn.Tanh()
		self.fusion = Fusion()
		self.lin11 = nn.Linear(fv_features, mid_features)
		self.lin12 = nn.Linear(fs_features, mid_features)
		self.lin2 = nn.Linear(mid_features, out_features)
		self.bn = nn.BatchNorm1d(mid_features)
		self.bn2 = nn.BatchNorm1d(mid_features)

	def forward(self, x, y):
		x = self.tanh(self.bn(self.lin11(self.drop(x)))) 
		y = self.tanh(self.bn2(self.lin12(self.drop(y))))
		x = self.lin2(self.drop(x + y))
		return x

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

		if config.pretrained_model == 'bert':
			self.bert = Bert(config.bert_model, config.max_question_len)
		elif config.pretrained_model == 'glove':
			self.embedding = Glove(embeddings, fine_tune=False)
		else:
			self.embedding = nn.Embedding(
				embeddings, embedding_features) # , padding_idx=0)
			nn.init.xavier_uniform_(self.embedding.weight)

	def _init_gru(self, weight):
		for w in weight.chunk(3, 0):
			nn.init.xavier_uniform_(w)

	def forward(self, q, q_len):
		if config.pretrained_model == 'bert':
			tanhed = self.bert.forward(q[0], q[1])
		else:
			embedded = self.embedding(q)
			tanhed = self.tanh(self.drop(embedded))
		packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
		_, h = self.gru(packed)
		return h.squeeze(0)

class FactExtractor(nn.Sequential):
	def __init__(self, embeddings):
		super(FactExtractor, self).__init__()
		self.add_module('embedding',  Glove(embeddings, fine_tune=False))

class Attention(nn.Module):
	def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
		super(Attention, self).__init__()
		self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  
		self.q_lin = nn.Linear(q_features, mid_features)
		self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

		self.drop = nn.Dropout(drop)
		self.relu = nn.ReLU(inplace=True)
		self.fusion = Fusion()

	def forward(self, v, q):
		v = self.v_conv(self.drop(v))
		q = self.q_lin(self.drop(q))
		q = tile_2d_over_nd(q, v)
		x = self.fusion(v, q)
		x = self.x_conv(self.drop(x))
		return x


def apply_attention(input, attention):
	""" Apply any number of attention maps over the input.
		The attention map has to have the same size in all dimensions except dim=1.
		n--batch_size, c--feature_size, s--regions
	"""
	n, c = input.size()[:2]
	glimpses = attention.size(1)

	# flatten the spatial dims into the third dim, since we don't need to 
	# care about how they are arranged
	input = input.view(n, c, -1)
	attention = attention.view(n, glimpses, -1)
	s = input.size(2)

	# apply a softmax to each attention map separately
	# since softmax only takes 2d inputs, we have to collapse the first two dimensions together
	# so that each glimpse is normalized separately
	attention = attention.view(n * glimpses, -1)
	attention = F.softmax(attention, dim=1)

	# apply the weighting by creating a new dim to tile both tensors over
	target_size = [n, glimpses, c, s]
	input = input.view(n, 1, c, s).expand(*target_size)
	attention = attention.view(n, glimpses, 1, s).expand(*target_size)
	weighted = input * attention
	# sum over only the spatial dimension
	weighted_mean = weighted.sum(dim=3, keepdim=True)
	# the shape at this point is (n, glimpses, c, 1)
	return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
	""" Repeat the same feature vector over all spatial positions of a given feature map.
		The feature vector should have the same batch size and number of features as the feature map.
	"""
	n, c = feature_vector.size()
	spatial_sizes = feature_map.size()[2:]
	# print('tile_2d_over_nd',feature_vector.shape,spatial_sizes)
	tiled = feature_vector.view(n, c, *([1] * len(spatial_sizes))).expand(n, c, *spatial_sizes)
	# print(tiled.shape)
	return tiled
