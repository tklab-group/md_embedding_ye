#!/bin/sh
python3 python1.py tomcat 1000 5000 1 2022_1_12_subword 20 700 128 SUB_WORD 1 1 0 > ./output/python1.txt &
python3 python2.py > ./output/python2.txt &
python3 python3.py > ./output/python3.txt &
python3 python4.py > ./output/python4.txt &
wait
python3 python_end.py
