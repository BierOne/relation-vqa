import os, sys
import json
import bcolz
import itertools
import numpy as np
from collections import Counter
sys.path.append(os.getcwd())
import utils.utils as utils
import utils.data as data
import detector.utils.config as config


def _get_file_(train=False, val=False, test=False, question=False, answer=False, version=config.version, qa_path=config.qa_path):
	""" Get the correct question or answer file."""
	_file = utils.path_for(train=train, val=val, test=test, 
							question=question, answer=answer, version=version, qa_path=qa_path)
	with open(_file, 'r') as fd:
		_object = json.load(fd)
	return _object

def load_json():
	""" Load json data and split relations into three sub elements. """
	relation_result = os.path.join(
				config.raw_data_path, 'vqa-rel_map_result_0.30.json')
	with open(relation_result, 'r') as fd:
		relation_result = json.load(fd)
	# rel_df = pd.DataFrame(relation_result)

	(image_id, question, question_id, relation, 
	answer, relation_id, score,
	relation_sub, relation_rel, relation_obj) = ([] for _ in range(10))
	for result in relation_result:
		for key_name in result:
			eval(key_name).append(result[key_name])
		r = result['relation'].split(', ', 2)
		relation_sub.append(r[0])
		relation_rel.append(r[1])
		relation_obj.append(r[2])

	assert len(set(question_id)) == len(relation_result)
	return (image_id, question_id, question, 
			relation, relation_id, 
			relation_sub, relation_rel, relation_obj)


def question_prepare(questions, top_k=None, start=0):
	""" Question vocab construction. """
	# print(questions)
	# print(questions[:500])
	# all_tokens = questions if top_k else itertools.chain.from_iterable(
										# map(utils.tokenize_text, questions))
	questions = list(map(utils.tokenize_text, questions))
	all_tokens = set(itertools.chain.from_iterable(questions))
	# print(list(all_tokens)[:500])
	
	vocab = {t: i for i, t in enumerate(all_tokens, start=0)} # leave 0 for <UNK>

	word_ids = [[vocab[t] for t in q] for q in questions]
	# print(word_ids[:5])
	# print(questions[:5])
	# print([vocab[t] for t in questions[0]])
	return	questions, word_ids, vocab


def filter_glove(question_vocab):
	""" Using the pre-trained glove vectors to initialize word embeddings.
		Filtering glove file and reshape the glove feature file so that the 
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
												word, np.zeros(300,)))

	embeddings = bcolz.carray(
		glove_weights_filtered[1:].reshape(len(question_vocab), 300),
		rootdir=config.glove_path_filtered,
		mode='w')
	embeddings.flush()


def most_frequent(sub_topk, rel_topk, obj_topk,
				relation_subs, relation_rels, relation_objs):
	""" Filtering the topk relation elements and replace less frequent 
		element with '<UNK>' (i.e., 0). 
	"""

	sub_counter = Counter(relation_subs).most_common(sub_topk)
	rel_counter = Counter(relation_rels).most_common(rel_topk)
	obj_counter = Counter(relation_objs).most_common(obj_topk)
	print(len(sub_counter), len(rel_counter), len(obj_counter))
	sub_vocab = {t[0]: i for i, t in enumerate(sub_counter, start=1)}
	rel_vocab = {t[0]: i for i, t in enumerate(rel_counter, start=1)}
	obj_vocab = {t[0]: i for i, t in enumerate(obj_counter, start=1)}

	sub_ids = [sub_vocab.get(i, 0) for i in relation_subs]
	rel_ids = [rel_vocab.get(i, 0) for i in relation_rels]	
	obj_ids = [obj_vocab.get(i, 0) for i in relation_objs]

	# remove samples that all three elements are removed
	remove_idxs = []
	for i, rel in enumerate(zip(sub_ids, rel_ids, obj_ids)):
		if rel[0] == rel[1] == rel[2] == 0:
			remove_idxs.append(i)
			
	return (sub_ids, rel_ids, obj_ids, remove_idxs, 
			sub_vocab, rel_vocab, obj_vocab)


def merge_data(relation_subs, relation_rels, relation_objs):
	""" Merge ambiguous elements. """
	alias_map = os.path.join(
				config.raw_data_path, 'alias_map_dict.json') 
	with open(alias_map, 'r') as fd:
		alias_map = json.load(fd)
	object_alias = alias_map['object_alias']
	relation_alias = alias_map['relation_alias']

	def update_alias(rel_class, alias_type):
		for ele in rel_class:
			if ele in alias_type:
				yield alias_type[ele]
			else:
				yield ele

	relation_subs = list(update_alias(relation_subs, object_alias))
	relation_rels = list(update_alias(relation_rels, relation_alias))
	relation_objs = list(update_alias(relation_objs, object_alias))

	return relation_subs, relation_rels, relation_objs


def main():
	(image_ids, question_ids, questions_rvqa, 
	relations, relation_ids,
	relation_subs, relation_rels, relation_objs) = load_json()
	
	questions_train = _get_file_(train=True, question=True)
	questions_val = _get_file_(val=True, question=True)
	questions_test = _get_file_(test=True, question=True)
	
	vqa2_qa_path = config.main_path+'../vqa/vqa2.0/qa_path/'
	v2_questions_train = _get_file_(train=True, question=True, version='v2', qa_path=vqa2_qa_path)
	v2_questions_val = _get_file_(val=True, question=True, version='v2', qa_path=vqa2_qa_path)
	v2_questions_test = _get_file_(test=True, question=True, version='v2', qa_path=vqa2_qa_path)
	
	questions_rvqa = list(data.prepare_questions(questions_rvqa, rvqa=True))
	
	questions_train = list(data.prepare_questions(questions_train))
	questions_val = list(data.prepare_questions(questions_val))
	questions_test = list(data.prepare_questions(questions_test))
	
	v2_questions_train = list(data.prepare_questions(v2_questions_train))
	v2_questions_val = list(data.prepare_questions(v2_questions_val))
	v2_questions_test = list(data.prepare_questions(v2_questions_test))

	questions = questions_rvqa + questions_train + questions_val + questions_test + v2_questions_train + v2_questions_val + v2_questions_test
	questions, word_ids, question_vocab = question_prepare(questions)
	filter_glove(question_vocab)
	print(len(question_vocab))
	
	if config.merge:
		relation_subs, relation_rels, relation_objs = merge_data(
						relation_subs, relation_rels, relation_objs)

	(sub_ids, rel_ids, obj_ids, remove_idxs,
	sub_vocab, rel_vocab, obj_vocab) = most_frequent(
		2000, 256, 2000, relation_subs, relation_rels, relation_objs)

	# the question_id can be used to locate sample
	meta_data = {q_id:{
		'image_id': image_ids[i], 
		'question': questions[i], 'word_id': word_ids[i],
		'relation': relations[i], 'relation_id': relation_ids[i],
		'relation_sub': relation_subs[i], 'sub_id': sub_ids[i],
		'relation_rel': relation_rels[i], 'rel_id': rel_ids[i],
		'relation_obj': relation_objs[i], 'obj_id': obj_ids[i]
		} for (i, q_id) in enumerate(question_ids) if q_id not in remove_idxs
	}
	
	vocabs = {
		'question': question_vocab,
		'subs': sub_vocab,
		'rels': rel_vocab,
		'objs': obj_vocab
	}

	with open(config.meta_data_path, 'w') as fd:
		json.dump(meta_data, fd)
	with open(config.vocab_path, 'w') as fd:
		json.dump(vocabs, fd)


if __name__ == '__main__':
	main()
