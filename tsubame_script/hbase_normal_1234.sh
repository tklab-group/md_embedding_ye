#!/bin/sh
#$ -l f_node=1
#$ -cwd
#$ -l h_rt=24:00:00
#$ -o hbase_normal_1234_out.txt
#$ -e hbase_normal_1234_err.txt
#$ -m be
#$ -M saberjunn@gmail.com
. /etc/profile.d/modules.sh
module load python
pip3 install -r ../requirements.txt
echo `date`
nvidia-smi
python3 single_project_run.py hbase 1000 5000 1 2022_1_14_normal 10 700 256 NORMAL 1 1 0 > ./output/2022_1_14_normal_hbase_group_1.txt &
python3 single_project_run.py hbase 1000 5000 2 2022_1_14_normal 10 700 256 NORMAL 1 1 1 > ./output/2022_1_14_normal_hbase_group_2.txt &
python3 single_project_run.py hbase 1000 5000 3 2022_1_14_normal 10 700 256 NORMAL 1 1 2 > ./output/2022_1_14_normal_hbase_group_3.txt &
python3 single_project_run.py hbase 1000 5000 4 2022_1_14_normal 10 700 256 NORMAL 1 1 3 > ./output/2022_1_14_normal_hbase_group_4.txt &
wait
echo 'end'
echo `date`