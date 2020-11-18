#!/bin/bash

conda activate robust-bdd

DATAFOLDER = $1 # Local data folder
TABLE = $2 #The number of the table in the paper
MODEL = $3 #The Model to reproduce, either twostream-resnet50 or dualhrnet
VARIANT = $4 #The Model variant: one of plain, swa ,multiadabn, multiadabnswa

cd $MODEL

if [ $TABLE -eq 1 ]; then
    python test.py "table_${TABLE}_${VARIANT}" --weights "paper" --data_folder $DATAFOLDER --mode "hold"
elif [ $TABLE -eq 2 ]; then
    python test.py "table_${TABLE}_${VARIANT}" --weights "paper" --data_folder $DATAFOLDER
elif [ $TABLE -eq 3 ]; then
    python test.py "table_${TABLE}_${VARIANT}_1" --weights "paper" --data_folder $DATAFOLDER
    python test.py "table_${TABLE}_${VARIANT}_2" --weights "paper" --data_folder $DATAFOLDER
    python test.py "table_${TABLE}_${VARIANT}_3" --weights "paper" --data_folder $DATAFOLDER
elif [ $TABLE -eq 4 ]; then
    python test.py "table_${TABLE}_${VARIANT}_g" --weights "paper" --data_folder $DATAFOLDER
    python test.py "table_${TABLE}_${VARIANT}_1" --weights "paper" --data_folder $DATAFOLDER
    python test.py "table_${TABLE}_${VARIANT}_2" --weights "paper" --data_folder $DATAFOLDER
    python test.py "table_${TABLE}_${VARIANT}_3" --weights "paper" --data_folder $DATAFOLDER