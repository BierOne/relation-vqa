# relation-vqa
Re-implementation for 'R-VQA: Learning Visual Relation Facts with Semantic Attention for Visual Question Answering'.

The paper was published on SIGKDD 2018 and can be downloaded at this [link](http://www.kdd.org/kdd2018/accepted-papers/view/r-vqa-learning-visual-relation-facts-with-semantic-attention-for-visual-que).

This repository focuses on the implementation of the Relation Fact Detector, and the later part (i.e., VQA with facts) will be finished shortly.
There are some slightly difference between this repository and the original paper:
* The image feature used here is extracted from faster RCNN, instead of the region-based CNN models. However, you can easily change the input 
with the later features.
* You can use pre-trained glove features to initialized word embeddings.
* I filtered the topk relations on the whole set, instead of the train+val splits.
* The samples with all the three elements (i.e., sub, rel, obj) are replaced with 'UNK' after the filtering will be removed.
* Other optimization methods (e.g., Adam), activation function (e.g., ReLU) and batch norm tricks are applied.

I greatly appreciate the first author Pan Lu for his help and the detailed reply to my questions!

## Prerequisites
pytorch==1.0.1  nltk==3.4  bcolz==1.2.1  h5py==2.9.0

## Dataset
The R-VQA dataset can be downloaded at Pan Lu's [repository](https://github.com/lupantech/rvqa). 

The VQA dataset can be downloaded at the 
[official website](https://visualqa.org/download.html). This repository only implemented the model on VQA 1.0 and 2.0 datasets. If you want to 
recover the results on COCO QA dataset, you need to write your own pytorch Dataset.

The pre-trained Glove features can be found on [glove website](https://nlp.stanford.edu/projects/glove/).

I guess you may need the RCNN image feature of Visual Genome, no hesitate to email me or drop me a message.

## Runing Details of Relation Fact Detector
Put all the raw data according to config.py.

1. Preprocess image features (Most times can skip this step and ask me for the h5 file).
```
python preprocess/preprocess_image_grid.py (or preprocess_image_rcnn.py if you use rcnn features)
```
2. Preprocess all the metadata. This will result all the needed meta json file and vocab files.
```
python preprocess/preprocess_meta.py
```
3. Train the detector.
```
python relation-fact-detector/main.py
```
4. Extract facts and vocabs.
```
python preprocess/preprocess_fact.py
python preprocess/preprocess_vocab.py
```
4. Train the RelAtt model.
```
python main.py
```


