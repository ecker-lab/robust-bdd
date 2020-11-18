#!/bin/bash

conda activate robust-bdd

#DATAFOLDER = $1 # Local data folder
#TABLE = $2 #The number of the table in the paper
#MODEL = $3 #The Model to reproduce, either twostream-resnet50 or dualhrnet
#VARIANT = $4 #The Model variant: one of plain, swa ,multiadabn, multiadabnswa

cd $3

if [ $2 -eq 1 ]; then
    python test.py "table_${2}_${4}" --weights "paper" --data_folder $1 --mode "hold"
elif [ $2 -eq 2 ]; then
    python test.py "table_${2}_${4}" --weights "paper" --data_folder $1
elif [ $2 -eq 3 ]; then
    python test.py "table_${2}_${4}_1" --weights "paper" --data_folder $1
    python test.py "table_${2}_${4}_2" --weights "paper" --data_folder $1
    python test.py "table_${2}_${4}_3" --weights "paper" --data_folder $1
elif [ $2 -eq 4 ]; then
    python test.py "table_${2}_${4}_g" --weights "paper" --data_folder $1
    python test.py "table_${2}_${4}_1" --weights "paper" --data_folder $1
    python test.py "table_${2}_${4}_2" --weights "paper" --data_folder $1
    python test.py "table_${2}_${4}_3" --weights "paper" --data_folder $1
fi