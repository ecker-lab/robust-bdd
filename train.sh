#!/bin/bash

conda activate robust-bdd

#DATAFOLDER = $1 # Local data folder
#TABLE = $2 #The number of the table in the paper
#MODEL = $3 #The Model to reproduce, either twostream-resnet50 or dualhrnet
#VARIANT = $4 #The Model variant: one of plain, multi

cd $3

if [ $2 -le 2 ]; then
    python train.py "table_${2}_${4}" --data_folder $1
elif [ $2 -eq 3 ]; then
    python train.py "table_${2}_${4}_1" --data_folder $1
    python train.py "table_${2}_${4}_2" --data_folder $1
    python train.py "table_${2}_${4}_3" --data_folder $1
elif [ $2 -eq 4 ]; then
    python train.py "table_${2}_${4}_g" --data_folder $1
    python train.py "table_${2}_${4}_1" --data_folder $1
    python train.py "table_${2}_${4}_2" --data_folder $1
    python train.py "table_${2}_${4}_3" --data_folder $1