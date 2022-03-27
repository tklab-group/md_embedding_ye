#!/bin/sh
python3 python1.py > ./output/python1.txt &
python3 python2.py > ./output/python2.txt &
python3 python3.py > ./output/python3.txt &
python3 python4.py > ./output/python4.txt &
wait
python3 python_end.py
