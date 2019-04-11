import os, sys
import json
import itertools
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



def main():
	# For question processing, we have done in preprocess_meta
	# process answers subject to fair training
	if config.train_set == 'train':
		answers = _get_file_(train=True, answer=True)
		answers = list(data.prepare_mul_answers(answers))
	else: # train+val
		answers = []
		for train in [True, False]:
			ans = _get_file_(train=train, val=not train, answer=True)
			answers += list(data.prepare_mul_answers(ans))

	answer_vocab = extract_vocab(answers, top_k=config.max_answers)
	
	with open(config.vocab_path, 'w+') as fd:
		vocabs = json.load(fd)
		vocabs['answer'] = answer_vocab
		json.dump(vocabs, fd)


if __name__ == '__main__':
	main()
