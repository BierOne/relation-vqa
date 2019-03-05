import os, sys
import json
import bcolz
import itertools
import numpy as np

import config
import utils


def load_json():
	""" Load json data and split relations into three sub elements. """
	relation_result = os.path.join(config.data_path, 'vqa-rel_map_result_0.30.json') 
	with open(relation_result, 'r') as fd:
		relation_result = json.load(fd)
	# r_df = pd.DataFrame(relation_result)

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
	print(len(set(relation_rel)))
	return (image_id, question_id, question, 
			relation, relation_id, 
			relation_sub, relation_rel, relation_obj)


def question_prepare(questions):
	""" Question vocab construction. """
	questions = [utils.process_questions(q) for q in questions]
	all_tokens = set(itertools.chain.from_iterable(questions))
	vocab = {t: i for i, t in enumerate(all_tokens)}
	word_ids = [[vocab[t] for t in q] for q in questions]
	return questions, word_ids, vocab


def relation_ele_prepare(relation_eles):
	""" Relation elements vocab construction. """
	relation_eles_unique = set(relation_eles)
	vocab = {t: i for i, t in enumerate(relation_eles_unique)}
	relation_eles_ids = [vocab[e] for e in relation_eles]
	return relation_eles_ids


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
						word, np.random.normal(scale=0.6, size=(300,))))

	embeddings = bcolz.carray(
		glove_weights_filtered[1:].reshape(len(question_vocab), 300),
		rootdir=config.glove_path_filtered,
		mode='w')
	embeddings.flush()


def split_data(dataset_len):
	""" Split the data with 60% train, 20% dev, 20% test. """
	train_ids, dev_ids, test_ids = [], [], []
	np.random.seed(6)
	idxs = np.random.choice(3, dataset_len, p=[0.6, 0.2, 0.2])

	for i, idx in enumerate(idxs):
		if idx == 0:
			train_ids.append(i)
		elif idx == 1:
			dev_ids.append(i)
		else:
			test_ids.append(i)
	return train_ids, dev_ids, test_ids


def main():
	(image_ids, questions, question_ids, 
	relations, relation_ids,
	relation_subs, relation_rels, relation_objs) = load_json()

	questions, word_ids, question_vocab = question_prepare(questions)
	filter_glove(question_vocab)

	relation_sub_ids = relation_ele_prepare(relation_subs)
	relation_rel_ids = relation_ele_prepare(relation_rels)
	relation_obj_ids = relation_ele_prepare(relation_objs)

	train_ids, dev_ids, test_ids = split_data(len(image_ids))

	meta_data = {
		'image_ids': image_ids, 'question_ids': question_ids,
		'questions': questions, 'word_ids': word_ids,
		'relations': relations, 'relation_ids': relation_ids,
		'relation_subs': relation_subs, 'relation_sub_ids': relation_sub_ids,
		'relation_rels': relation_rels, 'relation_rel_ids': relation_rel_ids,
		'relation_objs': relation_objs, 'relation_obj_ids': relation_obj_ids,
		'train_ids': train_ids, 'dev_ids': dev_ids, 'test_ids': test_ids,
	}
	with open(config.meta_data_path, 'w') as fd:
		json.dump(meta_data, fd)


if __name__ == '__main__':
	main()
