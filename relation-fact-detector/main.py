import os, sys
import argparse
import json, bcolz
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

import config
import data
import utils
import model


def run(net, loader, tracker, optimizer, 
                scheduler=None, loss_criterion=None, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        recall_1, recall_5, recall_10 = [], [], []

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=7)

    loss_t_all = tracker.track('{}_loss_all'.format(prefix), tracker_class(**tracker_params))
    loss_t_sub = tracker.track('{}_loss_sub'.format(prefix), tracker_class(**tracker_params))
    loss_t_rel = tracker.track('{}_loss_rel'.format(prefix), tracker_class(**tracker_params))
    loss_t_obj = tracker.track('{}_loss_obj'.format(prefix), tracker_class(**tracker_params))

    acc_t_sub = tracker.track('{}_acc_sub'.format(prefix), tracker_class(**tracker_params))
    acc_t_rel = tracker.track('{}_acc_rel'.format(prefix), tracker_class(**tracker_params))
    acc_t_obj = tracker.track('{}_acc_obj'.format(prefix), tracker_class(**tracker_params))

    for idx, v, q, rel, rel_sub, rel_rel, rel_obj, q_len in loader:
        v = v.cuda(async=True)
        q = q.cuda(async=True)
        rel_sub = rel_sub.cuda(async=True)
        rel_rel = rel_rel.cuda(async=True)
        rel_obj = rel_obj.cuda(async=True)
        q_len = q_len.cuda(async=True)

        sub_prob, rel_prob, obj_prob = net(v, q, rel_sub, rel_rel, rel_obj, q_len)

        loss_sub = loss_criterion(sub_prob, rel_sub)
        loss_rel = loss_criterion(rel_prob, rel_rel)
        loss_obj = loss_criterion(obj_prob, rel_obj)
        loss_all = config.lamda_sub*loss_sub + config.lamda_rel*loss_rel + config.lamda_obj*loss_obj

        acc_sub = utils.batch_accuracy(sub_prob, rel_sub).cpu()
        acc_rel = utils.batch_accuracy(rel_prob, rel_rel).cpu()
        acc_obj = utils.batch_accuracy(obj_prob, rel_obj).cpu()

        if train:
            # scheduler.step()
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
        else:
            # store information about evaluation of this minibatch
            recall_1.append(utils.batch_recall())
            recall_5.append(utils.batch_recall())
            recall_10.append(utils.batch_recall())
        
        loss_t_all.append(loss_all.item())
        loss_t_sub.append(loss_sub.item())
        loss_t_rel.append(loss_rel.item())
        loss_t_obj.append(loss_obj.item())

        acc_t_sub.append(acc_sub.mean())
        acc_t_rel.append(acc_rel.mean())
        acc_t_obj.append(acc_obj.mean())

        fmt = '{:.4f}'.format
        loader.set_postfix(
            loss_all=fmt(loss_t_all.mean.value),
            loss_sub=fmt(loss_t_sub.mean.value), acc_sub=fmt(acc_t_sub.mean.value),
            loss_rel=fmt(loss_t_rel.mean.value), acc_sub=fmt(acc_t_rel.mean.value),
            loss_obj=fmt(loss_t_obj.mean.value), acc_sub=fmt(acc_t_obj.mean.value)
        )

    if not train:
        recall_1 = torch.cat(recall_1, dim=0).numpy().mean()
        recall_5 = torch.cat(recall_5, dim=0).numpy().mean()
        recall_10 = torch.cat(recall_10, dim=0).numpy().mean()
        return recall_1, recall_5, recall_10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='saved and resumed file name')
    parser.add_argument('--resume', action='store_true', help='resumed flag')
    parser.add_argument('--test', dest='test_only', action='store_true')
    parser.add_argument('--gpu', default='0', help='the chosen gpu id')
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    ########################################## ARGUMENT SETTING  #####################################
    if args.test_only:
        args.resume = True
    if args.resume and not args.name:
        raise ValueError('Resuming requires file name!')
    name = args.name if args.name else datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.resume:
        target_name = name
        logs = torch.load(target_name)
    else: 
        target_name = os.path.join('logs', '{}'.format(name))
    if not args.test_only:
        print('will save to {}'.format(target_name))

    ######################################### DATASET PREPARATION #################################### 
    if args.test_only:
        val_loader = data.get_loader(split='test')
    else:
        train_loader = data.get_loader(split='train')
        if config.train_set == 'train':
            val_loader = data.get_loader(split='val')
        if config.train_set == 'train+val':
            val_loader = data.get_loader(split='test')

    ########################################## MODEL PREPARATION #####################################
    embeddings = bcolz.open(config.glove_path_filtered)[:]
    net = model.Net(embeddings).cuda()
    loss = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / 50000))
    optimizer = optim.RMSprop(
        [p for p in net.parameters() if p.requires_grad], 
        lr=config.initial_lr,
        momentum=0.98,
        weight_decay=0.01
    )

    start_epoch = 0
    if args.resume:
        net.load_state_dict(logs['weights'])
        start_epoch = logs['epoch']

    tracker = utils.Tracker()
    recall_10_val_best = 0.0

    for i in range(start_epoch, config.epochs):
        if not args.test_only:
            run(net, train_loader, tracker, optimizer, scheduler, 
                                loss_tracker=loss, train=True, prefix='train', epoch=i)
        if config.train_set=='train' or (
            not config.train_set == 'train' and i in range(config.epochs-5, config.epochs)):
            r = run(net, val_loader, tracker, optimizer, scheduler, 
                                loss_tracker=loss, train=False, prefix='val', epoch=i)
            print("Valid epoch {}: recall@1 is {}, recall@5 is {}, recall@10 is".format(
                                                                        i, r[0], r[1], r[2]))

        if not args.test_only:
            results = {
                'epoch': i,
                'name': name,
                'weights': net.state_dict()
            }
            if config.train_set == 'train' and r[2] > recall_10_val_best:
                recall_10_val_best = r[2]
                torch.save(results, target_name+'.pth')
            if not config.train_set == 'train':
                torch.save(results, target_name+'{}.pth')
                if config.train_set == 'train+val' and i in range(config.epochs-5, config.epochs):
                    print("Testing epoch {}: recall@1 is {}, recall@5 is {}, recall@10 is".format(
                                                                            i, r[0], r[1], r[2]))
        else:
            saved_for_test(val_loader, r)
            break


if __name__ == '__main__':
    main()
