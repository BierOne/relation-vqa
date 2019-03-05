import json
import h5py
import numpy as np
import torch
import torch.utils.data as data

import config


def get_loader(split):
	""" Returns a data loader for the desired split """   
	splits = FactData(split)
	loader = torch.utils.data.DataLoader(
		splits,
		batch_size=config.batch_size,
		shuffle=True if split=='train' else False,  # only shuffle the data in training
		pin_memory=True,
		num_workers=config.data_workers,
		collate_fn=collate_fn,
	)
	return loader

def collate_fn(batch):
	# put question lengths in descending order so that we can use packed sequences later
	batch.sort(key=lambda x: x[-1], reverse=True)
	return data.dataloader.default_collate(batch)


class FactData(data.Dataset):
	""" Relation Fact Detector dataset. """
	def __init__(self, split):
		super(FactData, self).__init__()
		with open(config.meta_data_path, 'r') as fd:
			self.meta_data = json.load(fd)

		# choose right split idxs
		if split == 'train':
			if config.train_set == 'train':
				self.splits = self.meta_data['train_ids']
			elif config.train_set == 'train+val':
				self.splits = self.meta_data['train_ids'] +self.meta_data['dev_ids']
			else: # all the index should be used
				self.splits = [i for i in range(len(self.meta_data['image_ids']))]
		if split == 'val':
			self.splits = self.meta_data['dev_ids']
		if split == 'test':
			self.splits = self.meta_data['test_ids']

		# image
		self.image_features_path = config.image_features_path
		self.vg_id_to_index = self._create_vg_id_to_index()

		# question
		self.questions = self.meta_data['word_ids']
		self.questions = [self._encode_question(q) for q in self.questions]

	def _encode_question(self, question):
		""" Turn a question into a vector of indices and a question length """
		vec = torch.zeros(config.max_question_len).long()
		for i, token in enumerate(question):
			if i < config.max_question_len:
				vec[i] = token
		return vec, min(len(question), config.max_question_len)

	def _create_vg_id_to_index(self):
		""" Create a mapping from a VG image id into the corresponding index into the h5 file """
		with h5py.File(self.image_features_path, 'r') as features_file:
			vg_ids = features_file['ids'][()]
		vg_id_to_index = {id: i for i, id in enumerate(vg_ids)}
		return vg_id_to_index

	def _load_image(self, image_id):
		""" Load an image from disk. """
		if not hasattr(self, 'features_file'):
			# Loading the h5 file has to be done here and not in __init__ because when the DataLoader
			# forks for multiple works, every child would use the same file object and fail.
			# Having multiple readers using different file objects is fine though, so we just init in here.
			self.features_file = h5py.File(self.image_features_path, 'r')
		index = self.vg_id_to_index[image_id]
		img = self.features_file['features'][index]
		boxes = self.features_file['boxes'][index]
		return torch.from_numpy(img), torch.from_numpy(boxes)

	def __getitem__(self, item):
		idx = self.splits[item]

		image_id = int(self.meta_data['image_ids'][idx])
		question, question_len = self.questions[idx]
		relation_id = int(self.meta_data['relation_ids'][idx])
		relation_sub_id = self.meta_data['relation_sub_ids'][idx]
		relation_rel_id = self.meta_data['relation_rel_ids'][idx]
		relation_obj_id = self.meta_data['relation_obj_ids'][idx]

		v, b = self._load_image(image_id)
		return (idx, v, question, relation_id, 
				relation_sub_id, relation_rel_id, relation_obj_id, question_len)

	def __len__(self):
		return len(self.splits)
