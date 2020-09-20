#!/bin/sh
nohup /usr/bin/time -v python src/nachos.py -model_out models/uop_th100_pmi00001.dill -docmin 1 -threshold 100 -pmi-threshold 0.00001 -nodisc -so -sym -coref longest -model unordered_pmi -file_list data/file_list_nyt_train -v 5000 -k 50 -cloze_file data/cloze_tests/nyt_test >uop_th100_pmi00001_test.log 2>&1 &
