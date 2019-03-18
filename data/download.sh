#!/bin/sh

# vqa v1.0
# questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip
# answers
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip


# vqa v2.0
# questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
# answers
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

# bottom up features (https://github.com/peteanderson80/bottom-up-attention)
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip https://imagecaption.blob.core.windows.net/imagecaption/test2015.zip
# alternative bottom-up features: 36 fixed proposals per image instead of 10--100 adaptive proposals per image.
#wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip

unzip "*.zip"
