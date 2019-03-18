# dataset version (v1 or v2)
version = 'v1'

# training set ('train' or 'train+val')
train_set = 'train'

# paths
main_path = '/raid/guoyangyang/vqa/'
# data_path = '/home/guoyangyang/project/vqa/' + 'rcnn-data/'
data_path = main_path + 'rcnn-data/'
if version == 'v1':
	qa_path = main_path+'vqa1.0/qa_path/'  # directory containing the question and annotation jsons from vqa v1.0 dataset
	fact_path = main_path + '../lyb/vqa1.0_facts/'
else:
	qa_path = main_path+'vqa2.0/qa_path/'  # directory containing the question and annotation jsons from vqa v2.0 dataset
	fact_path = main_path + '../lyb/vqa2.0_facts/'
bottom_up_trainval_path = main_path + 'rcnn-feature/trainval'  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = main_path + 'rcnn-feature/test2015'  # directory containing the .tsv file(s) with bottom up features
preprocessed_trainval_path = data_path + 'genome-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
preprocessed_test_path = data_path + 'genome-test.h5'  # path where preprocessed features from the test split are saved to and loaded from
vqa_vocabulary_path = fact_path + 'vocabs.json'  # path where the used vocabularies for question and answers are saved to
detector_path = 'detector/logs/'
detector_glove_path_filtered = main_path + '../lyb/detector/glove_filter'
fact_vocab_path = main_path + '../lyb/r-vqa/vocabs.json'
# import settings
task = 'OpenEnded'
dataset = 'mscoco'
max_question_len = 15
fact_length = 3
test_split = 'test2015'  # either 'test-dev2015' or 'test2015'

# text pretrained model config
pretrained_model = 'glove' # 'bert' or 'glove'
bert_model = 'bert-base-uncased'
# glove_path = main_path + 'word_embed/glove/'
# glove_path_filtered = glove_path + 'glove_filter'
if pretrained_model == 'bert':
	text_embed_size = 768
else: 
	text_embed_size = 300
fact_embed_size = 900
# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal

# training config
epochs = 100
batch_size = 200
initial_lr = 3e-4
glimpse = 1
data_workers = 4
max_answers = 3000
