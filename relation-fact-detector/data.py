import os, json
import h5py
import torch
import torch.utils.data as data

import config as config


def get_loader(split, meta_data):
	""" Returns a data loader for the desired split """	  
	splits = FactData(split, meta_data)
	loader = torch.utils.data.DataLoader(
		splits,
		batch_size=config.batch_size,
		shuffle=True if split=='train' else False,	# only shuffle the data in training
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
	def __init__(self, split, meta_data):
		super(FactData, self).__init__()
		self.meta_data = meta_data

		# choose right split path
		if split == 'train':
			if config.train_set == 'train':
				splits_path = os.path.join(
						config.raw_data_path, 'vqa_raw_train_0.30.json')
			elif config.train_set == 'train+val':
				splits_path = os.path.join(
						config.raw_data_path, 'vqa_raw_train_val_0.30.json')
			else: # all the index should be used
				splits_path = os.path.join(
						config.raw_data_path, 'vqa-rel_map_result_0.30.json')
		if split == 'val':
			splits_path = os.path.join(
						config.raw_data_path, 'vqa_raw_val_0.30.json')
		if split == 'test':
			splits_path = os.path.join(
						config.raw_data_path, 'vqa_raw_test_0.30.json')

		with open(splits_path, 'r') as fd:
			splits = json.load(fd)
		self.question_ids = [i['question_id'] for i in splits]
		self.question_ids = [q_id 
			for q_id in self.question_ids if q_id in self.meta_data]

		# image
		self.image_features_path = config.image_features_path
		self.vg_id_to_index = self._create_vg_id_to_index()

		# question
		self.questions = [self._encode_question(
			self.meta_data[q_id]['word_id']) for q_id in self.question_ids]

	def _encode_question(self, question):
		""" Turn a question into a vector of indices and a question length """
		vec = torch.zeros(config.max_question_len).long()
		for i, token in enumerate(question):
			if i < config.max_question_len:
				vec[i] = token
		return vec, min(len(question), config.max_question_len)

	def _encode_answers(self, answers):
		""" Turn an answer into a vector """
		# answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
		# this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
		# to get the loss that is weighted by how many humans gave that answer
		answer_vec = torch.zeros(len(self.answer_to_index))
		for answer in answers:
			index = self.answer_to_index.get(answer)
			if index is not None:
				answer_vec[index] += 1
		return answer_vec

	def _create_vg_id_to_index(self):
		""" Create a mapping from a VG image id into the 
			orresponding index into the h5 file """
		with h5py.File(self.image_features_path, 'r') as features_file:
			vg_ids = features_file['ids'][()]
		vg_id_to_index = {id: i for i, id in enumerate(vg_ids)}
		return vg_id_to_index

	def _load_image(self, image_id):
		""" Load an image from disk. """
		if not hasattr(self, 'features_file'):
			self.features_file = h5py.File(self.image_features_path, 'r')
		index = self.vg_id_to_index[image_id]
		img = self.features_file['features'][index]
		boxes = self.features_file['boxes'][index]
		return torch.from_numpy(img), torch.from_numpy(boxes)

	def __getitem__(self, item):
		q_id = self.question_ids[item]
		sample = self.meta_data[q_id]

		image_id = int(sample['image_id'])
		q, q_len = self.questions[item]
		r_id = sample['relation_id']
		r_sub_id = sample['sub_id']
		r_rel_id = sample['rel_id']
		r_obj_id = sample['obj_id']

		v, b = self._load_image(image_id)
		return v, q, r_id, r_sub_id, r_rel_id, r_obj_id, q_len

	def __len__(self):
		return len(self.question_ids)
