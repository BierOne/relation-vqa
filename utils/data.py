import os, re
import json, h5py
from PIL import Image

import torch
import torch.utils.data as data

import utils.config as config
import utils.utils as utils
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
	trainval_path = config.rcnn_trainval_path if config.image_feature == 'rcnn' \
				else config.grid_trainval_path
	test_path = config.rcnn_test_path if config.image_feature == 'rcnn' \
				else config.grid_test_path
	split = VQA(
		utils.path_for(train=train, val=val, test=test, question=True),
		utils.path_for(train=train, val=val, test=test, answer=True),
		utils.path_for(train=train, val=val, test=test, knowledge=True),
		trainval_path if not test else test_path,
		answerable_only=train,
		dummy_answers=test,
	)
	if do_val_later:
		val = True
		train = False
		split += VQA(
			utils.path_for(train=train, val=val, test=test, question=True),
			utils.path_for(train=train, val=val, test=test, answer=True),
			utils.path_for(train=train, val=val, test=test, knowledge=True),
			trainval_path if not test else test_path,
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
	def __init__(self, questions_path, answers_path, knowledge_path, image_features_path,
								answerable_only=False, dummy_answers=False):
		super(VQA, self).__init__()

		with open(questions_path, 'r') as fd:
			questions_json = json.load(fd)
		with open(answers_path, 'r') as fd:
			answers_json = json.load(fd)
		# with open(fact_path, 'r') as fd:
			# facts_json = json.load(fd)
		if preloaded_vocab:
			vocab_json = preloaded_vocab
		else:
			with open(config.vocab_path, 'r') as fd:
				vocab_json = json.load(fd)

		# self._check_integrity(questions_json, answers_json)
		self.question_ids = [q['question_id'] for q in questions_json['questions']]
		
		# self.facts = torch.tensor([facts_json[str(qid)]['fact_index'] for qid in self.question_ids])
		if not dummy_answers:
			self.question_types = [a['question_type'] for a in answers_json['annotations']]
			# self.invalid_type = ['how many', 'is the', 'is this', 'is this a', \
				# 'is there a', 'is it', 'is there', 'does the', 'are these', 'are there',\
				# 'is', 'is the man' , 'are', 'does this', 'how many people are', 'do', \
				# 'are they', 'what time', 'are there any', 'is he', 'is the woman', \
				# 'is this an', 'do you', 'how many people are in', 'has', 'is this person', \
				# 'can you', 'is the person', 'could', 'was', 'is that a', 'what number is']
			self.invalid_type = []
			self.knowledges = self._encode_knowledge(knowledge_path)
		# vocab
		self.vocab = vocab_json
		self.token_to_index = self.vocab['question']
		self.answer_to_index = self.vocab['answer']

		# q and a
		self.questions = list(prepare_questions(questions_json))
		self.questions = [self._encode_question(utils.tokenize_text(q)) for q in self.questions]

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

	def _encode_knowledge(self, knowledge_path, topk=10):
		""" Encode all the knowledge extracted from r-vqa here. """
		with h5py.File(knowledge_path, 'r') as fd:
			q_id_dict = {q_id: idx for idx, q_id in enumerate(fd['qids'])}
			# print(len(q_id_dict), len(self.question_ids))
			knowledges = []
			for q_id in self.question_ids:
				idx = q_id_dict[q_id]
				if self.question_types[idx] not in self.invalid_type:
					knowledges.append(
						(torch.from_numpy(fd['subs'][idx][:topk]),
						torch.from_numpy(fd['rels'][idx][:topk]),
						torch.from_numpy(fd['objs'][idx][:topk])))
				else:
					knowledges.append((torch.zeros(topk, dtype=torch.int64), 
										torch.zeros(topk, dtype=torch.int64), 
										torch.zeros(topk, dtype=torch.int64)))
			return knowledges
			
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
		f = torch.tensor(self.knowledges[item]) # f -> 10x3
		image_id = self.coco_ids[item]
		v, b = self._load_image(image_id)
		# since batches are re-ordered for PackedSequence's, the original question order is lost
		# we return `item` so that the order of (v, q, a) triples can be restored if desired
		# without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
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


class CocoImages(data.Dataset):
	""" Dataset for MSCOCO images located in a folder on the filesystem """
	def __init__(self, path, transform=None):
		super(CocoImages, self).__init__()
		self.path = path
		self.id_to_filename = self._find_images()
		self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
		print('found {} images in {}'.format(len(self), self.path))
		self.transform = transform

	def _find_images(self):
		id_to_filename = {}
		for filename in os.listdir(self.path):
			if not filename.endswith('.jpg'):
				continue
			id_and_extension = filename.split('_')[-1]
			id = int(id_and_extension.split('.')[0])
			id_to_filename[id] = filename
		return id_to_filename

	def __getitem__(self, item):
		id = self.sorted_ids[item]
		path = os.path.join(self.path, self.id_to_filename[id])
		img = Image.open(path).convert('RGB')

		if self.transform is not None:
			img = self.transform(img)
		return id, img

	def __len__(self):
		return len(self.sorted_ids)


class Composite(data.Dataset):
	""" Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
	def __init__(self, *datasets):
		self.datasets = datasets

	def __getitem__(self, item):
		current = self.datasets[0]
		for d in self.datasets:
			if item < len(d):
				return d[item]
			item -= len(d)
		else:
			raise IndexError('Index too large for composite dataset')

	def __len__(self):
		return sum(map(len, self.datasets))
