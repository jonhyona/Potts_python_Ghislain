#! /bin/bash
#SBATCH --partition=wide2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=ghislain.delabbey@sissa.it

module load python
rm data_analysis/*.pkl
for g in 0. 0.5 1.
do
    python create_pkl_file.py 0 $g 100 &
done
wait
for g in 0. 0.5 1.
do
    for cue in {0..199}
    do
	python -u run.py $cue 1. 100 &
    done
done
wait
