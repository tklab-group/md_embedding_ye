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
python3 tomcat_normal_group_1.py > ./output/tomcat_normal_group_1.txt &
python3 tomcat_normal_group_2.py > ./output/tomcat_normal_group_2.txt &
python3 tomcat_normal_group_3.py > ./output/tomcat_normal_group_3.txt &
python3 tomcat_normal_group_4.py > ./output/tomcat_normal_group_4.txt &
wait
echo 'end'
echo `date`