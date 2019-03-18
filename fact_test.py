import json,os
import utils.utils as utils
import utils.config as config
import h5py

to_print = 100
qa_path = utils.path_for(train=False, val=True, test=False, question=True)
fact_path = utils.path_for(train=False, val=True, test=False, fact=True)
with open(fact_path) as fd:
	fact_json = json.load(fd)
with open(qa_path) as fd:
	q_json = json.load(fd)
questions = {}
for q in q_json['questions']:
	questions[q['question_id']] = (q['question'],q['image_id'])
with open(config.fact_vocab_path, 'r') as fd:
	fact_vocab_json = json.load(fd)
sub_index_to_string = {i: s for s, i in fact_vocab_json['subs'].items()}
rel_index_to_string = {i: s for s, i in fact_vocab_json['rels'].items()}
obj_index_to_string = {i: s for s, i in fact_vocab_json['objs'].items()}
(sub_index_to_string[0], rel_index_to_string[0], obj_index_to_string[0]) = ('unk', 'unk', 'unk')
# questions = [(q['question_id'], q['question']) for q in questions_json['questions']]

result_file = '{}_{}2014_facts'.format(config.dataset, 'val')
result_file = os.path.join(config.fact_path, result_file)
with h5py.File('{}.hdf5'.format(result_file),"r") as f:
	subs = f['subs']
	objs = f['objs']
	rels = f['rels']
	qids = f['qids']
	for idx, qid in enumerate(qids):
		fact = fact_json[str(qid)]
		item = questions[qid]
		print('qid: {}, question: {}, img_id: {}'.format(qid, item[0], item[1]))
		print(fact['fact_in_string'][:])
		print(sub_index_to_string[subs[idx,0]], rel_index_to_string[rels[idx,0]], 		obj_index_to_string[objs[idx,0]], "\n")
		if idx > to_print:
			exit()