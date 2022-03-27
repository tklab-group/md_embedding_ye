#!/bin/sh
#$ -l f_node=1
#$ -cwd
#$ -l h_rt=24:00:00
#$ -o many_normal_1_out.txt
#$ -e many_normal_1_err.txt
#$ -m be
#$ -M saberjunn@gmail.com
. /etc/profile.d/modules.sh
module load python
pip3 install -r ../requirements.txt
echo `date`
nvidia-smi
python3 single_project_run.py hadoop 1000 5000 6 2022_1_14_many_normal 10 700 256 NORMAL 1 1 0 > ./output/2022_1_14_many_normal_hadoop.txt &
python3 single_project_run.py lucene 1000 5000 6 2022_1_14_many_normal 10 700 256 NORMAL 1 1 1 > ./output/2022_1_14_many_normal_lucene.txt &

wait
echo 'end'
echo `date`