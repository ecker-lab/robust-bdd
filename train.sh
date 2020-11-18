#!/bin/bash

conda activate robust-bdd

DATAFOLDER = $1 # Local data folder
TABLE = $2 #The number of the table in the paper
MODEL = $3 #The Model to reproduce, either twostream-resnet50 or dualhrnet
VARIANT = $4 #The Model variant: one of plain, multi

cd $MODEL

if [ $TABLE -le 2 ]; then
    python train.py "table_${TABLE}_${VARIANT}" --data_folder $DATAFOLDER
elif [ $TABLE -eq 3 ]; then
    python train.py "table_${TABLE}_${VARIANT}_1" --data_folder $DATAFOLDER
    python train.py "table_${TABLE}_${VARIANT}_2" --data_folder $DATAFOLDER
    python train.py "table_${TABLE}_${VARIANT}_3" --data_folder $DATAFOLDER
elif [ $TABLE -eq 4 ]; then
    python train.py "table_${TABLE}_${VARIANT}_g" --data_folder $DATAFOLDER
    python train.py "table_${TABLE}_${VARIANT}_1" --data_folder $DATAFOLDER
    python train.py "table_${TABLE}_${VARIANT}_2" --data_folder $DATAFOLDER
    python train.py "table_${TABLE}_${VARIANT}_3" --data_folder $DATAFOLDER