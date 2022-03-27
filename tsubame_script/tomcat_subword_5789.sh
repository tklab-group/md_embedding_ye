#!/bin/sh
#$ -l f_node=1
#$ -cwd
#$ -l h_rt=24:00:00
#$ -o out.txt
#$ -e err.txt
#$ -m be
#$ -M saberjunn@gmail.com
. /etc/profile.d/modules.sh
module load python
pip3 install -r ../requirements.txt
echo `date`
nvidia-smi
python3 single_project_run.py tomcat 1000 5000 5 2022_1_12_subword 20 700 128 SUB_WORD 1 1 0 > ./output/tomcat_subword_group_1.txt &
python3 single_project_run.py tomcat 1000 5000 7 2022_1_12_subword 20 700 128 SUB_WORD 1 1 1 > ./output/tomcat_subword_group_2.txt &
python3 single_project_run.py tomcat 1000 5000 8 2022_1_12_subword 20 700 128 SUB_WORD 1 1 2 > ./output/tomcat_subword_group_3.txt &
python3 single_project_run.py tomcat 1000 5000 9 2022_1_12_subword 20 700 128 SUB_WORD 1 1 3 > ./output/tomcat_subword_group_4.txt &
wait
echo 'end'
echo `date`