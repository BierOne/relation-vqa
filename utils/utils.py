import os
from nltk.tokenize import TweetTokenizer
import utils.config as config


def batch_accuracy(predicted, true):
	""" Compute the accuracies for a batch of predictions and answers """
	_, predicted_index = predicted.max(dim=1, keepdim=True)
	agreeing = true.gather(dim=1, index=predicted_index)

	return (agreeing * 0.3).clamp(max=1)


def path_for(train=False, val=False, test=False, question=False, answer=False, fact=False, version=config.version, qa_path=config.qa_path, test_split=config.test_split):
	assert train + val + test == 1
	assert question + answer + fact == 1
	
	if train:
		split = 'train2014'
	elif val:
		split = 'val2014'
	else:
		split = test_split

	if question:
		fmt = '{0}_{1}_{2}_questions.json'
	elif fact:
		fmt = '{1}_{2}_facts.json'
	else:
		if test:
			# just load validation data in the test=answer=True case, will be ignored anyway
			split = 'val2014'
		fmt = '{1}_{2}_annotations.json'

	if version == 'v2':
		fmt = 'v2_' + fmt
	s = fmt.format(config.task, config.dataset, split)
	if not fact:
		return os.path.join(qa_path, s)
	else:
		return os.path.join(config.fact_path, s)


def tokenize_text(text):
	tknzr = TweetTokenizer(preserve_case=False)
	return tknzr.tokenize(text)

	
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
