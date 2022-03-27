#!/bin/sh
#$ -l f_node=1
#$ -cwd
#$ -l h_rt=24:00:00
#$ -o many_subword_1_out.txt
#$ -e many_subword_1_err.txt
#$ -m be
#$ -M saberjunn@gmail.com
. /etc/profile.d/modules.sh
module load python
pip3 install -r ../requirements.txt
echo `date`
nvidia-smi
python3 single_project_run.py hadoop 1000 5000 6 2022_1_14_many_subword 20 700 128 SUB_WORD 1 1 0 > ./output/2022_1_14_many_subword_hadoop.txt &
python3 single_project_run.py lucene 1000 5000 6 2022_1_14_many_subword 20 700 128 SUB_WORD 1 1 1 > ./output/2022_1_14_many_subword_lucene.txt &

wait
echo 'end'
echo `date`