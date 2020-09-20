#!/bin/sh
nohup /usr/bin/time -v python src/nachos.py -model_out models/bg_th100_pmi00001_sym.dill -docmin 1 -threshold 100 -pmi-threshold 0.00001 -nodisc -so -coref longest -model bigram -file_list data/file_list_nyt_train -v 5000 -k 50 -cloze_file data/cloze_tests/nyt_test >bg_th100_pmi00001_sym_test.log 2>&1 &
