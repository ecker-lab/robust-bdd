#!/bin/bash

conda activate robust-bdd

TABLE = $1 #The number of the table in the paper
MODEL = $2 #The Model to reproduce, either twostream-resnet50 or dualhrnet
VARIANT = $3 #The Model variant: one of plain, swa, classicadabn, classicadabnswa, multiadabn, multiadabnswa
WEIGHTS = $4 #The weights to use: either a path to a weights file or paper, in which case the weights used in the paper are automatically downloaded (if not there allready) and then used.