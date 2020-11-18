#!/bin/bash

conda create --name robust-bdd python=3.7
conda init bash
conda activate robust-bdd
pip3 install --upgrade pip 
pip3 install numpy==1.19.0
pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 torchcontrib==0.0.2
pip3 install matplotlib==3.2.2 tqdm Pillow shapely==1.7.0 opencv-python==4.2.0.34 pandas==1.0.5 scikit-learn==0.23.1 imgaug==0.4.0 imantics==0.1.12 scipy==1.5.0 scikit-image==0.17.2 yacs
pip3 install --upgrade setuptools


DATA_FOLDER = $1

python twostream-resnet50/utils/make_dataset.py --data-folder $DATAFOLDER/train/ --output-folder $DATAFOLDER/resize256/ --mode manymulti --cutwidth 256 --resizewidth 256
python twostream-resnet50/utils/make_dataset.py --data-folder $DATAFOLDER/tier3/ --output-folder $DATAFOLDER/resize256_3/ --mode manymulti --cutwidth 256 --resizewidth 256
python twostream-resnet50/utils/make_dataset.py --data-folder $DATAFOLDER/test/ --output-folder $DATAFOLDER/test256/ --mode manymulti --cutwidth 256 --resizewidth 256
python twostream-resnet50/utils/make_dataset.py --data-folder $DATAFOLDER/hold/ --output-folder $DATAFOLDER/hold/all/ --mode nocut

conda deactivate