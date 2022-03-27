#!/bin/sh
#$ -l f_node=1
#$ -cwd
#$ -l h_rt=24:00:00
#$ -o cassandra_subword_5789_out.txt
#$ -e cassandra_subword_5789_err.txt
#$ -m be
#$ -M saberjunn@gmail.com
. /etc/profile.d/modules.sh
module load python
pip3 install -r ../requirements.txt
echo `date`
nvidia-smi
python3 single_project_run.py cassandra 1000 5000 5 2022_1_17_subword 20 700 128 SUB_WORD 1 1 0 > ./output/cassandra_subword_group_5.txt &
python3 single_project_run.py cassandra 1000 5000 7 2022_1_17_subword 20 700 128 SUB_WORD 1 1 1 > ./output/cassandra_subword_group_7.txt &
python3 single_project_run.py cassandra 1000 5000 8 2022_1_17_subword 20 700 128 SUB_WORD 1 1 2 > ./output/cassandra_subword_group_8.txt &
python3 single_project_run.py cassandra 1000 5000 9 2022_1_17_subword 20 700 128 SUB_WORD 1 1 3 > ./output/cassandra_subword_group_9.txt &
wait
echo 'end'
echo `date`