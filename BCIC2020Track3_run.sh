#!/bin/bash

# if you only have a single GPU, you can run the following command
# python3 BCIC2020Track3_train.py --gpu 0 --folds "0-15"

python3 BCIC2020Track3_train.py --gpu 0 --folds "0-7"   &
python3 BCIC2020Track3_train.py --gpu 1 --folds "7-15"  &
wait

