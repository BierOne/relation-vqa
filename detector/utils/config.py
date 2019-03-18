# training set ('train', 'train+val' or 'all')
train_set = 'all'
version = 'v1'
merge = True # merge relations or not

# paths
main_path = '/raid/guoyangyang/lyb/'
if version == 'v1':
	qa_path = main_path+'../vqa/vqa1.0/qa_path/'  # directory containing the question and annotation jsons from vqa v1.0 dataset
else:
	qa_path = main_path+'../vqa/vqa2.0/qa_path/'  # directory containing the question and annotation jsons from vqa v2.0 dataset

data_path = main_path + 'genome_feature/'
raw_data_path = main_path + 'r-vqa/'
meta_data_path = raw_data_path + 'meta_data.json'
vocab_path = raw_data_path + 'vocabs.json'
bottom_up_path = main_path + 'VG100K'
image_features_path = data_path + 'VG.h5'
glove_path = main_path + '../vqa/word_embed/glove/'
glove_path_filtered = main_path + 'detector/glove_filter'
# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal

# hyper parameters
max_question_len = 15
text_embed_size = 300
lamda_sub = 2.0 #1.0
lamda_rel = 0.8 #0.8
lamda_obj = 1.5 #1.2

# training config
epochs = 100
batch_size = 100
initial_lr = 3e-4
data_workers = 3

