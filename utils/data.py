import os, re
import json, h5py
import numpy as np

import torch
import torch.utils.data as data

import utils.config as config
import utils.utils as utils
from model.pretrained_models import Bert
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


preloaded_vocab = None
# monkey-patch ConcatDataset so that it delegates member access, e.g. VQA(...).num_tokens
data.ConcatDataset.__getattr__ = lambda self, attr: getattr(self.datasets[0], attr)

def get_loader(train=False, val=False, test=False):
	""" Returns a data loader for the desired split """
	if train and val:
		do_val_later = True
		val = False
	else:
		do_val_later = False
	split = VQA(
		utils.path_for(train=train, val=val, test=test, question=True),
		utils.path_for(train=train, val=val, test=test, answer=True),
		utils.path_for(train=train, val=val, test=test, fact=True),
		config.preprocessed_trainval_path if not test else config.preprocessed_test_path,
		answerable_only=train,
		dummy_answers=test,
	)
	if do_val_later:
		val = True
		train = False
		split += VQA(
			utils.path_for(train=train, val=val, test=test, question=True),
			utils.path_for(train=train, val=val, test=test, answer=True),
			utils.path_for(train=train, val=val, test=test, fact=True),
			config.preprocessed_trainval_path if not test else config.preprocessed_test_path,
			answerable_only=val,
			dummy_answers=test,
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
	def __init__(self, questions_path, answers_path, fact_path, image_features_path,
								answerable_only=False, dummy_answers=False):
		super(VQA, self).__init__()

		with open(questions_path, 'r') as fd:
			questions_json = json.load(fd)
		with open(answers_path, 'r') as fd:
			answers_json = json.load(fd)
		with open(fact_path, 'r') as fd:
			facts_json = json.load(fd)
		if preloaded_vocab:
			vocab_json = preloaded_vocab
		else:
			with open(config.vqa_vocabulary_path, 'r') as fd:
				vocab_json = json.load(fd)

		# self._check_integrity(questions_json, answers_json)
		self.question_ids = [q['question_id'] for q in questions_json['questions']]
		print(len(self.question_ids), len(facts_json))
		self.facts = torch.tensor([facts_json[str(qid)]['fact_index'] for qid in self.question_ids])
		# vocab
		self.vocab = vocab_json
		self.token_to_index = self.vocab['question']
		self.answer_to_index = self.vocab['answer']

		# q and a
		self.questions = list(prepare_questions(questions_json))
		if config.pretrained_model == 'bert':
			bert_model = Bert(config.bert_model, config.max_question_len, True)
			self.questions = [bert_model.tokenize_text(q) for q in self.questions]
		else:
			self.questions = [self._encode_question(
							utils.tokenize_text(q)) for q in self.questions]

		self.answers = list(prepare_answers(answers_json))
		self.answers = [self._encode_answers(a) for a in self.answers]

		# v
		self.image_features_path = image_features_path
		self.coco_id_to_index = self._create_coco_id_to_index()
		self.coco_ids = [q['image_id'] for q in questions_json['questions']]

		self.dummy_answers= dummy_answers

		# only use questions that have at least one answer
		self.answerable_only = answerable_only
		if self.answerable_only:
			self.answerable = self._find_answerable()

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

	def _find_answerable(self):
		""" Create a list of indices into questions that will have at least one answer that is in the vocab """
		answerable = []
		for i, answers in enumerate(self.answers):
			# store the indices of anything that is answerable
			answer_has_index = len(answers.nonzero()) > 0
			if answer_has_index:
				answerable.append(i)
		return answerable

	def _encode_question(self, question):
		""" Turn a question into a vector of indices and a question length """
		vec = torch.zeros(config.max_question_len).long()
		for i, token in enumerate(question):
			if i < config.max_question_len:
				index = self.token_to_index[token]
				vec[i] = index
		return vec, min(len(question), config.max_question_len)

	def _encode_answers(self, answers):
		""" Turn an answer into a vector """
		# answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
		# this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
		# to get the loss that is weighted by how many humans gave that answer
		answer_vec = torch.zeros(len(self.answer_to_index))
		for answer in answers:
			index = self.answer_to_index.get(answer)
			if index is not None: # all zero
				answer_vec[index] += 1
		return answer_vec

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
		return torch.from_numpy(img).unsqueeze(1), torch.from_numpy(boxes)

	def __getitem__(self, item):
		if self.answerable_only:
			item = self.answerable[item]
		if not self.dummy_answers:
			a = self.answers[item]
		else:
			# just return a dummy answer, it's not going to be used anyway
			a = 0
		f = self.facts[item] # f -> 10x3
		image_id = self.coco_ids[item]
		v, b = self._load_image(image_id)
		# since batches are re-ordered for PackedSequence's, the original question order is lost
		# we return `item` so that the order of (v, q, a) triples can be restored if desired
		# without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
		if config.pretrained_model == 'bert':
			q_ids, q_mask = self.questions[item]
			return item, v, q_ids, q_mask, a, b, f, config.max_question_len+2
		else:
			q, q_len = self.questions[item]
			return item, v, q, q, a, b, f, q_len

	def __len__(self):
		if self.answerable_only:
			return len(self.answerable)
		else:
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


dummy_vqa = lambda: None
dummy_vqa.getQuesIds = lambda: None
vqa_eval = VQAEval(dummy_vqa, None)

def process_answers(answer):
	answer = answer.replace('\n', ' ')
	answer = answer.replace('\t', ' ')
	answer = answer.strip()
	answer = vqa_eval.processPunctuation(answer)
	answer = vqa_eval.processDigitArticle(answer)
	return answer

def prepare_answers(answers_json):
	""" Normalize answers from a given answer json in the usual VQA format. """
	answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
	for answer_list in answers:
		yield list(map(process_answers, answer_list))

def prepare_mul_answers(answers_json):
	""" This can give more accurate answer selection. """
	answers = [ans_dict['multiple_choice_answer'] for ans_dict in answers_json['annotations']]
	return list(map(process_answers, answers)) 
