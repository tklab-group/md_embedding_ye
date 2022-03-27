#!/bin/sh
#$ -l f_node=1
#$ -cwd
#$ -l h_rt=00:10:00
#$ -o out.txt
#$ -e err.txt
. /etc/profile.d/modules.sh
module load python
pip3 install -r ../requirements.txt
python3 test_job.py
