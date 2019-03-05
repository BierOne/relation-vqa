# training set ('train', 'train+val' or 'all')
train_set = 'train'

# paths
data_path = '../data/'
glove_path = data_path
glove_path_filtered = glove_path + 'glove_filter'
meta_data_path = data_path + 'meta_data.json'
bottom_up_path = data_path + 'VG100K'
image_features_path = data_path + 'VG.h5'

# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal

# hyper parameters
max_question_len = 15
text_embed_size = 300
lamda_sub = 1.0
lamda_rel = 0.8
lamda_obj = 1.2

# training config
epochs = 100
batch_size = 100
initial_lr = 3e-4
data_workers = 1
