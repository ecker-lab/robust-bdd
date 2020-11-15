#!/bin/bash

conda activate robust-bdd

TABLE = $1 #The number of the table in the paper
MODEL = $2 #The Model to reproduce, either twostream-resnet50 or dualhrnet
VARIANT = $3 #The Model variant: one of plain, swa, classicadabn, classicadabnswa, multiadabn, multiadabnswa