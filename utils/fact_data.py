import os, re
import json, h5py
import torch
import torch.utils.data as data

import utils.config as config
import utils.utils as utils
from model.pretrained_models import Bert
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

preloaded_vocab = None
# monkey-patch ConcatDataset so that it delegates member access, e.g. VQA(...).num_tokens
data.ConcatDataset.__getattr__ = lambda self, attr: getattr(self.datasets[0], attr)

def get_loader(train=False, val=False, test=False, test_split=config.test_split ):
	""" Returns a data loader for the desired split """
	if train and val:
		do_val_later = True
		val = False
	else:
		do_val_later = False
	split = VQA(
		utils.path_for(train=train, val=val, test=test, question=True, test_split=test_split),
		config.preprocessed_trainval_path if not test else config.preprocessed_test_path,
	)
	if do_val_later:
		val = True
		train = False
		split += VQA(
			utils.path_for(train=train, val=val, test=test, question=True),
			config.preprocessed_trainval_path if not test else config.preprocessed_test_path,
		)
	loader = torch.utils.data.DataLoader(
		split,
		batch_size=config.batch_size,
		shuffle=train,	# only shuffle the data in training
		pin_memory=True,
		num_workers=config.data_workers,
		collate_fn=collate_fn,
	)
	return loader

def collate_fn(batch):
	# put question lengths in descending order so that we can use packed sequences later
	batch.sort(key=lambda x: x[-1], reverse=True)
	return data.dataloader.default_collate(batch)

class VQA(data.Dataset):
	""" VQA dataset, open-ended """
	def __init__(self, questions_path,  image_features_path,
								answerable_only=False, dummy_answers=False):
		super(VQA, self).__init__()

		with open(questions_path, 'r') as fd:
			questions_json = json.load(fd)

		with open(config.fact_vocab_path, 'r') as fd:
			vocab_json = json.load(fd)

		# self._check_integrity(questions_json, answers_json)
		self.question_ids = [q['question_id'] for q in questions_json['questions']]
		# vocab
		self.vocab = vocab_json
		self.token_to_index = self.vocab['question']
		# q and a
		self.questions = list(prepare_questions(questions_json))
		if config.pretrained_model == 'bert':
			bert_model = Bert(config.bert_model, config.max_question_len, True)
			self.questions = [bert_model.tokenize_text(q) for q in self.questions]
		else:
			self.questions = [self._encode_question(
							utils.tokenize_text(q)) for q in self.questions]
		# v
		self.image_features_path = image_features_path
		self.coco_id_to_index = self._create_coco_id_to_index()
		self.coco_ids = [q['image_id'] for q in questions_json['questions']]

	def _create_coco_id_to_index(self):
		""" Create a mapping from a COCO image id into the corresponding index into the h5 file """
		with h5py.File(self.image_features_path, 'r') as features_file:
			coco_ids = features_file['ids'][()]
		coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
		return coco_id_to_index

	def _check_integrity(self, questions, answers):
		""" Verify that we are using the correct data """
		qa_pairs = list(zip(questions['questions'], answers['annotations']))
		assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answers'
		assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
		assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
		assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'


	def _encode_question(self, question):
		""" Turn a question into a vector of indices and a question length """
		vec = torch.zeros(config.max_question_len).long()
		for i, token in enumerate(question):
			if i < config.max_question_len:
				index = self.token_to_index[token]
				vec[i] = index
		return vec, min(len(question), config.max_question_len)


	def _load_image(self, image_id):
		""" Load an image """
		if not hasattr(self, 'features_file'):
			# Loading the h5 file has to be done here and not in __init__ because when the DataLoader
			# forks for multiple works, every child would use the same file object and fail.
			# Having multiple readers using different file objects is fine though, so we just init in here.
			self.features_file = h5py.File(self.image_features_path, 'r')
		index = self.coco_id_to_index[image_id]
		img = self.features_file['features'][index]
		boxes = self.features_file['boxes'][index]
		return torch.from_numpy(img), torch.from_numpy(boxes)

	def __getitem__(self, item):
			# just return a dummy answer, it's not going to be used anyway
		image_id = self.coco_ids[item]
		v, b = self._load_image(image_id)
		if config.pretrained_model == 'bert':
			q_ids, q_mask = self.questions[item]
			return item, v, q_ids, q_mask, b,  config.max_question_len+2
		else:
			q, q_len = self.questions[item]
			return item, v, q, q, b,  q_len

	def __len__(self):
		return len(self.questions)

# this is used for normalizing questions
_special_chars = re.compile('(\'+s)*[^a-z0-9- ]*')

def prepare_questions(questions, rvqa=False):
	""" Tokenize and normalize questions from a given question json in the usual VQA format. """
	if not rvqa:
		questions = [q['question'] for q in questions['questions']]
	# print(questions[:10])
	for question in questions:
		question = question.lower()[:-1]
		question = _special_chars.sub('', question)
		question = re.sub(r'-+', ' ', question)
		yield question



