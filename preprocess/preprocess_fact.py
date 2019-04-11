import os, sys
import argparse
import json, bcolz
from tqdm import tqdm
from datetime import datetime

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
sys.path.append(os.getcwd())
import utils.config as config
import utils.fact_data as data
import utils.utils as utils
import detector.model.model as detector

def run(detector, loader, tracker, prefix='extract', epoch=0, top=10):
	""" Run an epoch over the given loader """
	# assert not (train and not has_answers)
	facts, idxs, subs, objs, rels = [], [], [], [], []
	fmt = '{:.4f}'.format
	tracker_class, tracker_params = tracker.MeanMonitor, {}
	loader = tqdm(loader, desc='{}'.format(prefix), ncols=0)
	unk_t_rel = tracker.track('{}_unk_rel'.format(prefix), tracker_class(**tracker_params))
	unk_t_sub = tracker.track('{}_unk_sub'.format(prefix), tracker_class(**tracker_params))
	unk_t_obj = tracker.track('{}_unk_obj'.format(prefix), tracker_class(**tracker_params))
	for idx, v, q, q_dummy, b, q_len in loader:
		v = v.cuda(async=True).squeeze(2)
		b = b.cuda(async=True)
		q = q.cuda(async=True)
		sub_prob, rel_prob, obj_prob = detector(v, q, 0, 0, 0, q_len)
		
		_, prob_sub_index = sub_prob.max(dim=1)
		unk_t_sub.append(torch.sum(prob_sub_index==0).item())
		
		_, prob_rel_index = rel_prob.max(dim=1)
		unk_t_rel.append(torch.sum(prob_rel_index==0).item())
		
		_, prob_obj_index = obj_prob.max(dim=1)
		unk_t_obj.append(torch.sum(prob_obj_index==0).item())
		
		top_10_sub = sub_prob.topk(top)[1]
		top_10_rel = rel_prob.topk(top)[1]
		top_10_obj = obj_prob.topk(top)[1]
		subs.append(top_10_sub.cpu())
		objs.append(top_10_obj.cpu())
		rels.append(top_10_rel.cpu())
		top_10_fact = torch.cat((top_10_sub, top_10_rel, top_10_obj), dim=1).view(-1,                        3,top).transpose(1,2)
		facts.append(top_10_fact.cpu())
		idxs.append(idx.view(-1).clone())
		
		loader.set_postfix(unk_sub=fmt(unk_t_sub.mean.value), 
		unk_rel=fmt(unk_t_rel.mean.value), unk_obj=fmt(unk_t_obj.mean.value) )
		# break
	facts = torch.cat(facts, dim=0).numpy()
	idxs = torch.cat(idxs, dim=0).numpy()
	subs = torch.cat(subs, dim=0).numpy()
	objs = torch.cat(objs, dim=0).numpy()
	rels = torch.cat(rels, dim=0).numpy()
	return idxs, subs, rels, objs, facts
def saved_for_test(test_loader, result, split='train'):
	""" in test mode, save a facts file in the format accepted by the submission server. """
	facts = {}
	with open(config.vocab_path, 'r') as fd:
		vocab_data = json.load(fd)
	sub_index_to_string = {i: s for s, i in vocab_data['subs'].items()}
	rel_index_to_string = {i: s for s, i in vocab_data['rels'].items()}
	obj_index_to_string = {i: s for s, i in vocab_data['objs'].items()}
	(sub_index_to_string[0], rel_index_to_string[0], obj_index_to_string[0]) = ('unk', 'unk', 'unk')
	question_ids = test_loader.dataset.question_ids
	# assert(len(result[0]) == len(question_ids))
	result_file = '{}_{}_facts'.format(config.dataset, split)
	if config.version == 'v2':
		result_file = 'v2_' + result_file
	result_file = os.path.join(config.fact_path, result_file)
	with h5py.File('{}.h5'.format(result_file),"w") as f:
		subs = f.create_dataset("subs",data=result[1])
		objs = f.create_dataset("objs",data=result[3])
		rels = f.create_dataset("rels",data=result[2])
		qids = [question_ids[i] for i in result[0]]
		qids = f.create_dataset("qids",data=qids)
		# print(subs)
		print('facts', result[4][1,1,:], subs[1], rels[1], objs[1])
		for fact, index, qid in zip(result[4], result[0], qids):
			assert(question_ids[index] == qid)
			fact_in_string = [( sub_index_to_string[f[0].item()], rel_index_to_string[f[1].item()],obj_index_to_string[f[2].item()]) for f in fact]
			facts[str(qid)] = {
			'fact_in_string':fact_in_string,
			'fact_index': fact.tolist()
			}
			# facts.append(entry)
	with open('{}.json'.format(result_file), 'w') as fd:
		json.dump(facts, fd)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', default='2019-03-16_10:28:52{}.pth', help='the name of detector')
	parser.add_argument('--gpu', default='6', help='the chosen gpu id')
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	cudnn.benchmark = True
	########################################## MODEL PREPARATION ########################################
	embedding = bcolz.open(config.glove_path_filtered)[:] 
	detector_path = os.path.join('detector/logs/',args.name) 
	detector_logs = torch.load(detector_path)
	detector_net = detector.Net(embedding).cuda()
	detector_net.load_state_dict(detector_logs['weights'])
	detector_net.eval()
	######################################### DATASET PREPARATION #######################################
	split = [ 'test-dev2015', 'train2014', 'val2014', 'test2015']
	# split = ['test-dev2015', 'test2015']
	
	tracker = utils.Tracker()
	for set in split:
		if set == 'train2014':
			val_loader = data.get_loader(train=True)
		elif set == 'val2014':
			val_loader = data.get_loader(val=True)
		elif set == 'trainval':
			val_loader = data.get_loader(train=True, val=True)
		else:
			val_loader = data.get_loader(test=True, test_split=set)
		
		facts = run(detector_net, val_loader, tracker, prefix=set, top=10)
		saved_for_test(val_loader, facts, set)
	
	
	
if __name__ == '__main__':
	main()