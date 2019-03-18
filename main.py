import os, sys
import argparse
import json, bcolz
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

import utils.config as config
import utils.data as data
import utils.utils as utils
import model.model as model
# import detector.model as detector

def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
	""" Run an epoch over the given loader """
	assert not (train and not has_answers)
	if train:
		net.train()
		tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
	else:
		net.eval()
		tracker_class, tracker_params = tracker.MeanMonitor, {}
		answ = []
		idxs = []
		accs = []

	loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
	loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
	acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

	for idx, v, q, q_dummy, a, b, f, q_len in loader:
		v = v.cuda(async=True)
		a = a.cuda(async=True)
		b = b.cuda(async=True)
		f = f.cuda(async=True)
		
		if config.pretrained_model == 'bert':
			q_ids = q.cuda(async=True)
			q_mask = q_dummy.cuda(async=True)
			q = (q_ids, q_mask)
		else:
			q = q.cuda(async=True)
		# sub_prob, rel_prob, obj_prob = detector(v.squeeze(2), q, 0, 0, 0, q_len)
		# top_10_sub = sub_prob.topk(10)[1]
		# top_10_rel = rel_prob.topk(10)[1]
		# top_10_obj = obj_prob.topk(10)[1]
		# top_10_fact = torch.cat((top_10_sub, top_10_rel, top_10_obj), dim=1).view(-1,                        3,10).transpose(1,2)
		# top_10_fact = torch.randint(500,(v.shape[0], 10, 3))
		out = net(v, b, q, f, q_len)
		if has_answers:
			# print(out.shape, a.shape)
			nll = -F.log_softmax(out, dim=1)
			loss = (nll * a / 10).sum(dim=1).mean()
			acc = utils.batch_accuracy(out, a).cpu()

		if train:
			scheduler.step()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		else:
			# store information about evaluation of this minibatch
			_, answer = out.cpu().max(dim=1)
			answ.append(answer.view(-1))
			if has_answers:
				accs.append(acc.view(-1))
			idxs.append(idx.view(-1).clone())

		if has_answers:
			loss_tracker.append(loss.item())
			for ac in acc:
				acc_tracker.append(ac.item())
			# acc_tracker.append(acc.mean())
			fmt = '{:.4f}'.format
			loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

	if not train:
		answ = torch.cat(answ, dim=0).numpy()
		if has_answers:
			accs = torch.cat(accs, dim=0).numpy()
		else:
			accs = []
		idxs = torch.cat(idxs, dim=0).numpy()
		return answ, accs, idxs


def saved_for_test(test_loader, result, epoch=None):
	""" in test mode, save a results file in the format accepted by the submission server. """
	answer_index_to_string = {a: s for s, a in test_loader.dataset.answer_to_index.items()}
	results = []
	for answer, index in zip(result[0], result[2]):
		answer = answer_index_to_string[answer.item()]
		qid = test_loader.dataset.question_ids[index]
		entry = {
			'question_id': qid,
			'answer': answer,
		}
		results.append(entry)
	result_file = 'vqa_{}_{}_{}_{}_{}_results.json'.format(
		config.task, config.dataset, config.test_split, config.version, epoch)
	with open(result_file, 'w') as fd:
		json.dump(results, fd)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='saved and resumed file name')
	parser.add_argument('--resume', action='store_true', help='resumed flag')
	parser.add_argument('--test', dest='test_only', default=False, action='store_true')
	parser.add_argument('--detctor', default='2019-03-16_10:28:52{}.pth', help='the name of detector')
	parser.add_argument('--gpu', default='7', help='the chosen gpu id')
	args = parser.parse_args()


	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	cudnn.benchmark = True

	########################################## ARGUMENT SETTING	 ########################################
	if args.test_only:
		args.resume = True
	if args.resume and not args.name:
		raise ValueError('Resuming requires file name!')
	name = args.name if args.name else datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	if args.resume:
		target_name = name
		logs = torch.load(target_name)
		# hacky way to tell the VQA classes that they should use the vocab without passing more params around
		data.preloaded_vocab = logs['vocab']
	else: 
		target_name = os.path.join('logs', '{}'.format(name))
	if not args.test_only:
		print('will save to {}'.format(target_name))



	######################################### DATASET PREPARATION #######################################
	if config.train_set == 'train':
		train_loader = data.get_loader(train=True)
		val_loader = data.get_loader(val=True)
	elif args.test_only:
		val_loader = data.get_loader(test=True)
	else:
		train_loader = data.get_loader(train=True, val=True)
		val_loader = data.get_loader(test=True)
	########################################## MODEL PREPARATION ########################################
	fact_embedding = bcolz.open(config.detector_glove_path_filtered)[:] 
	# detector_path = os.path.join(config.detector_path,args.detctor) 
	# detector_logs = torch.load(detector_path)
	# detector_net = detector.Net(fact_embedding).cuda()
	# detector_net.load_state_dict(detector_logs['weights'])
	# detector_net.eval()
	
	# if config.pretrained_model == 'glove':
		# embedding = bcolz.open(config.glove_path_filtered)[:] 
	# else:
		# embedding = len(val_loader.dataset.token_to_index)
	net = model.Net(fact_embedding)
	net = nn.DataParallel(net).cuda()
	
	optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], 
		lr=config.initial_lr,
		weight_decay=1e-8
	)

	# optimizer = optim.RMSprop(
		# [p for p in net.parameters() if p.requires_grad], 
		# lr=config.initial_lr,
		# momentum=0.20,
		# weight_decay=1e-8
	# )
	scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / 50000))
	######################################### 
	#######################################
	start_epoch = 0
	if args.resume:
		net.load_state_dict(logs['weights'])
		start_epoch = logs['epoch']

	tracker = utils.Tracker()
	acc_val_best = 0.0

	for i in range(start_epoch, config.epochs):
		if not args.test_only:
			run(net, train_loader, optimizer, scheduler, tracker, train=True, prefix='train', epoch=i)
		if not (config.train_set == 'train+val' and i in range(config.epochs-5)):
			r = run(net, val_loader, optimizer, scheduler, tracker, train=False, 
					prefix='val', epoch=i, has_answers=(config.train_set == 'train'))

		if not args.test_only:
			results = {
				'epoch': i,
				'name': name,
				'weights': net.state_dict(),
				'eval': {
					'answers': r[0],
					'accuracies': r[1],
					'idx': r[2]
				},
				'vocab': val_loader.dataset.vocab,
			}
			if config.train_set == 'train' and r[1].mean() > acc_val_best:
				acc_val_best = r[1].mean()
				torch.save(results, target_name+'.pth')
			if config.train_set == 'train+val':
				torch.save(results, target_name+'{}.pth')
				if i in range(config.epochs-5, config.epochs):
					saved_for_test(val_loader, r, i)
				
		else:
			saved_for_test(val_loader, r)
			break


if __name__ == '__main__':
	main()
