import re
import numpy as np
from nltk.tokenize import TweetTokenizer


def batch_accuracy(predicted, true):
	""" Compute the accuracies for a batch of predictions and truths. """
	_, predicted_index = predicted.max(dim=1)  
	return predicted_index, predicted_index.eq(true).float()


def recall(gt_sub, gt_rel, gt_obj, top_sub, top_rel, top_obj):
	""" The possibility of a predicted fact is the sum of probabilities
		of the subject, relation, and object. 
	"""
	r = []
	for idx in range(len(gt_sub)):
		if gt_sub[idx] in top_sub[idx] \
		and gt_rel[idx] in top_rel[idx] \
		and gt_obj[idx] in top_obj[idx]:
			r.append(1.0)
		else:
			r.append(0.0)
	return np.array(r).mean()


# this is used for normalizing questions
_special_chars = re.compile('(\'+s)*[^a-z0-9- ]*')

def process_questions(question):
	""" Tokenize and normalize questions from a given question json. """
	question = question.lower()[:-1]
	question = _special_chars.sub('', question)
	question = re.sub(r'-+', ' ', question)

	tnkzr = TweetTokenizer(preserve_case=False)
	return tnkzr.tokenize(question)


class Tracker:
	""" Keep track of results over time, while having access to monitors to display information about them. """
	def __init__(self):
		self.data = {}

	def track(self, name, *monitors):
		""" Track a set of results with given monitors under some name (e.g. 'val_acc').
			When appending to the returned list storage, use the monitors to retrieve useful information.
		"""
		l = Tracker.ListStorage(monitors)
		self.data.setdefault(name, []).append(l)
		return l

	def to_dict(self):
		# turn list storages into regular lists
		return {k: list(map(list, v)) for k, v in self.data.items()}


	class ListStorage:
		""" Storage of data points that updates the given monitors """
		def __init__(self, monitors=[]):
			self.data = []
			self.monitors = monitors
			for monitor in self.monitors:
				setattr(self, monitor.name, monitor)

		def append(self, item):
			for monitor in self.monitors:
				monitor.update(item)
			self.data.append(item)

		def __iter__(self):
			return iter(self.data)

	class MeanMonitor:
		""" Take the mean over the given values """
		name = 'mean'

		def __init__(self):
			self.n = 0
			self.total = 0

		def update(self, value):
			self.total += value
			self.n += 1

		@property
		def value(self):
			return self.total / self.n

	class MovingMeanMonitor:
		""" Take an exponentially moving mean over the given values """
		name = 'mean'

		def __init__(self, momentum=0.9):
			self.momentum = momentum
			self.first = True
			self.value = None

		def update(self, value):
			if self.first:
				self.value = value
				self.first = False
			else:
				m = self.momentum
				self.value = m * self.value + (1 - m) * value
