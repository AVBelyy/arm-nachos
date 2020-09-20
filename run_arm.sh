#!/bin/sh
nohup /usr/bin/time -v python src/nachos.py -model_out models/arm_lm.dill -docmin 1 -threshold 1 -so -coref longest -model arm -arm_vocab nyt_arm_vocab.txt -arm_rules gen_rules_nyt_100_0.00001_bin.txt -file_list data/file_list_nyt_train -v 5000 -k 50 -cloze_file data/cloze_tests/nyt_test >arm_lm_bin_test.log 2>&1 &
