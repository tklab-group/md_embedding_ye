#!/bin/sh
#$ -l f_node=1
#$ -cwd
#$ -l h_rt=24:00:00
#$ -o lucene_subword_1234_out.txt
#$ -e lucene_subword_1234_err.txt
#$ -m be
#$ -M saberjunn@gmail.com
. /etc/profile.d/modules.sh
module load python
pip3 install -r ../requirements.txt
echo `date`
nvidia-smi
python3 single_project_run.py lucene 1000 5000 1 2022_1_17_subword 20 700 128 SUB_WORD 1 1 0 > ./output/lucene_subword_group_1.txt &
python3 single_project_run.py lucene 1000 5000 2 2022_1_17_subword 20 700 128 SUB_WORD 1 1 1 > ./output/lucene_subword_group_2.txt &
python3 single_project_run.py lucene 1000 5000 3 2022_1_17_subword 20 700 128 SUB_WORD 1 1 2 > ./output/lucene_subword_group_3.txt &
python3 single_project_run.py lucene 1000 5000 4 2022_1_17_subword 20 700 128 SUB_WORD 1 1 3 > ./output/lucene_subword_group_4.txt &
wait
echo 'end'
echo `date`