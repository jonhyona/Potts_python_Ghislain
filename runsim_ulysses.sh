#! /bin/bash
#SBATCH --partition=regular2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mem=10000
#SBATCH --mail-user=ghislain.delabbey@sissa.it

rm *.pkl
rm *.txt
python create_pkl_file.py 0 0. 100 &
wait
for cue in {0..31}
do
    python run.py $cue 0. 100 &
done
wait
