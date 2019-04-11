# dataset version (v1 or v2)
version = 'v2'

# training set ('train' or 'train+val')
train_set = 'train+val'

# paths
main_path = '/raid/guoyangyang/vqa/'
rcnn_path = main_path + '/rcnn-data/'
grid_path = main_path + '/grid-data/'
img_path = main_path + 'mscoco'

if version == 'v1':
	qa_path = main_path+'vqa1.0/qa_path/'  # directory containing the question and annotation jsons from vqa v1.0 dataset
	fact_path = main_path + '../lyb/vqa1.0_facts/'
else:
	qa_path = main_path+'vqa2.0/qa_path/'  # directory containing the question and annotation jsons from vqa v2.0 dataset
	fact_path = main_path + '../lyb/vqa2.0_facts/'

bottom_up_trainval_path = main_path + 'rcnn-feature/trainval'  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = main_path + 'rcnn-feature/test2015'  # directory containing the .tsv file(s) with bottom up features
rcnn_trainval_path = rcnn_path + 'genome-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
rcnn_test_path = rcnn_path + 'genome-test.h5'  # path where preprocessed features from the test split are saved to and loaded from

image_train_path = main_path + 'mscoco/train2014'  # directory of training images
image_val_path = main_path + 'mscoco/val2014'  # directory of validation images
image_test_path = main_path + 'mscoco/test2015'  # directory of test images
grid_trainval_path = grid_path + 'resnet-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
grid_test_path = grid_path + 'resnet-test.h5'  # path where preprocessed features from the test split are saved to and loaded from


# dectector settings (preprocess [meta, question, answer])
merge = True # merge relations or not
raw_data_path = main_path + '../lyb/r-vqa/'
meta_data_path = raw_data_path + 'meta_data.json'
vocab_path = raw_data_path + 'vocabs.json' # path where the used vocabularies for question and answers and facts are saved to
glove_path = main_path + '../vqa/word_embed/glove/'
glove_path_filtered = raw_data_path + 'glove_filter'


# import settings
task = 'OpenEnded'
dataset = 'mscoco'
max_question_len = 15
fact_length = 3
test_split = 'test-dev2015'  # either 'test-dev2015' or 'test2015'

# text pretrained model config
pretrained_model = 'glove' # 'glove' or 'None'

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
