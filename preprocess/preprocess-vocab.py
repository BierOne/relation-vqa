import os, sys
import json
import itertools
import bcolz
import numpy as np 
from collections import Counter
sys.path.append(os.getcwd())

import utils.config as config
import utils.data as data
import utils.utils as utils


def _get_file_(train=False, val=False, test=False, question=False, answer=False):
	""" Get the correct question or answer file."""
	_file = utils.path_for(train=train, val=val, test=test, 
							question=question, answer=answer)
	with open(_file, 'r') as fd:
		_object = json.load(fd)
	return _object


def extract_vocab(iterable, top_k=None, start=0):
	""" Turns an iterable of list of tokens into a vocabulary.
		These tokens could be single answers or word tokens in questions.
	"""
	all_tokens = iterable if top_k else itertools.chain.from_iterable(
										map(utils.tokenize_text, iterable))
	counter = Counter(all_tokens)
	if top_k:
		most_common = counter.most_common(top_k)
		most_common = (t for t, c in most_common)
	else:
		most_common = counter.keys()
	# descending in count, then lexicographical order
	tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
	vocab = {t: i for i, t in enumerate(tokens, start=start)}
	return vocab


def filter_glove(question_vocab):
	""" Filtering glove file and reshape the glove feature file so that the 
		embedding features are all about the current question vocabulary.
	"""
	glove_file = os.path.join(config.glove_path, 'glove.6B.300d.txt')
	glove_weights = {}
	glove_weights_filtered = bcolz.carray(np.zeros(1))

	with open(glove_file, 'r') as g:
		for line in g:
			split = line.split()
			word = split[0]
			embedding = np.array([float(val) for val in split[1:]])
			glove_weights[word] = embedding

	# find words in glove which are from the current vocabulary
	for word in question_vocab:
		glove_weights_filtered.append(glove_weights.get(
												word, np.zeros(300)))
	   
	embeddings = bcolz.carray(
		glove_weights_filtered[1:].reshape(len(question_vocab), 300), # padding
		rootdir=config.glove_path_filtered,
		mode='w')
	embeddings.flush()


def main():
	# For question processing, we aim to use the pre-trained glove embedding
	# vectors, thus all questions are processed for filtering glove.
	# questions_train = _get_file_(train=True, question=True)
	# questions_val = _get_file_(val=True, question=True)
	# questions_test = _get_file_(test=True, question=True)

	# questions_train = list(data.prepare_questions(questions_train))
	# questions_val = list(data.prepare_questions(questions_val))
	# questions_test = list(data.prepare_questions(questions_test))

	# questions = questions_train + questions_val + questions_test

	# process answers subject to fair training
	if config.train_set == 'train':
		answers = _get_file_(train=True, answer=True)
		answers = list(data.prepare_mul_answers(answers))
	else: # train+val
		answers = []
		for train in [True, False]:
			ans = _get_file_(train=train, val=not train)
			answers += list(data.prepare_mul_answers(ans))

	answer_vocab = extract_vocab(answers, top_k=config.max_answers)
	# question_vocab = extract_vocab(questions, start=0)
	# filter_glove(question_vocab)
	
	with open(config.fact_vocab_path, 'r') as fd:
		fact_vocab = json.load(fd)
	vocabs = {
		# 'question': question_vocab,
		'question': fact_vocab['question'],
		'answer': answer_vocab,
	}
	with open(config.vqa_vocabulary_path, 'w') as fd:
		json.dump(vocabs, fd)


if __name__ == '__main__':
	main()
